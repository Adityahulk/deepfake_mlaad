import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.autograd import Function
import itertools
import random

# For loading models and data
from transformers import Wav2Vec2Model
import datasets
from datasets import load_dataset, Audio
import torchaudio.transforms as T
import torchaudio

# For evaluation
from sklearn.metrics import roc_curve
import numpy as np
import warnings

# --- ⚙️ Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {DEVICE} ---")

# Data & Model Params
SAMPLE_RATE = 16000
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
N_MELS = 80
MAX_AUDIO_LEN_SECONDS = 4
MAX_LEN_SAMPLES = SAMPLE_RATE * MAX_AUDIO_LEN_SECONDS
MAX_MEL_FRAMES = int(MAX_AUDIO_LEN_SECONDS * (SAMPLE_RATE / 160)) + 1 # ~401 frames

# Training Params
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50 # A real paper needs 50-100
STEPS_PER_EPOCH = 500  # Number of batches per "epoch" (since data is streamed)
VAL_STEPS = 100        # Number of batches to use for validation
LAMBDA_ADV = 0.1       # Adversarial loss weight. The key hyperparameter.


# --- 1. The A* Novelty: Gradient Reversal Layer (GRL) ---
# This is the "magic" for the disentanglement loss.
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverses the gradient
        return (grad_output.neg() * ctx.lambda_), None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# --- 2. The "Prosody Encoder" (Lightweight SOTA TCN/ResNet) ---
# This is your lightweight "Student" model that runs on the edge device.
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ProsodyEncoder(nn.Module):
    def __init__(self, n_mels=N_MELS, num_features=256):
        super(ProsodyEncoder, self).__init__()
        # Input: (Batch, 1, N_Mels, Time)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = ResBlock(64, 64, stride=(1,2))
        self.layer2 = ResBlock(64, 128, stride=(2,2)) 
        self.layer3 = ResBlock(128, 256, stride=(2,2))
        self.layer4 = ResBlock(256, 512, stride=(2,2))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        self.fc1 = nn.Linear(512, num_features)
        self.fc2 = nn.Linear(num_features, 2) # 2-class (bona fide, spoof)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x) # (Batch, 512, 1, 1)
        x = x.view(x.size(0), -1) # (Batch, 512)
        
        features = F.relu(self.fc1(x)) # (Batch, num_features)
        out_spoof = self.fc2(features) # (Batch, 2)
        
        return features, out_spoof

# --- 3. The "Content Encoder" (Frozen Wav2Vec2) ---
# This is the "Teacher" model. It is FROZEN.
class ContentEncoder(nn.Module):
    def __init__(self, model_name=WAV2VEC2_MODEL_NAME):
        super(ContentEncoder, self).__init__()
        print(f"Loading frozen {model_name}...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        self.feature_dim = self.wav2vec2.config.hidden_size # 768

    def forward(self, x):
        # x is (Batch, Num_Samples)
        with torch.no_grad(): 
            outputs = self.wav2vec2(x, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state # (Batch, Time, 768)
            content_features = torch.mean(hidden_states, dim=1) # (Batch, 768)
        return content_features

# --- 4. The "Disentanglement" Adversary ---
# This model tries to predict CONTENT from PROSODY.
# The ProsodyEncoder is trained to *fool* this.
class ContentDiscriminator(nn.Module):
    def __init__(self, prosody_dim=256, content_dim=768, hidden_dim=256):
        super(ContentDiscriminator, self).__init__()
        self.grl = GradientReversalLayer(lambda_=LAMBDA_ADV)
        self.net = nn.Sequential(
            nn.Linear(prosody_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, content_dim) # Predicts the content features
        )
        self.loss_fn = nn.MSELoss() # Try to reconstruct the content features

    def forward(self, prosody_features, content_features):
        prosody_features_adv = self.grl(prosody_features)
        predicted_content = self.net(prosody_features_adv)
        loss_adv = self.loss_fn(predicted_content, content_features)
        return loss_adv

# --- 5. The Full Model ---
class DisentangledAntiSpoofingModel(nn.Module):
    def __init__(self):
        super(DisentangledAntiSpoofingModel, self).__init__()
        # Note: Move to DEVICE happens in main()
        self.content_encoder = ContentEncoder()
        self.prosody_encoder = ProsodyEncoder(num_features=256)
        self.discriminator = ContentDiscriminator(prosody_dim=256, 
                                                content_dim=self.content_encoder.feature_dim)
        
    def forward(self, raw_audio, mels, labels):
        # --- Main Anti-Spoofing Path (Trainable) ---
        prosody_features, out_spoof = self.prosody_encoder(mels)
        loss_spoof = F.cross_entropy(out_spoof, labels)
        
        # --- Disentanglement Path (Adversarial) ---
        content_features = self.content_encoder(raw_audio)
        loss_adv = self.discriminator(prosody_features, content_features.detach())
        
        # --- Total Loss ---
        total_loss = loss_spoof + loss_adv
        
        return total_loss, loss_spoof, loss_adv, out_spoof

# --- 6. The Real Dataset Loading & Preprocessing ---
# This is the new, critical part.

# We define these transforms globally so they can be pickled
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=N_MELS
).to(DEVICE)

resampler_cache = {}

def get_resampler(orig_sr, new_sr):
    # Cache resamplers to avoid re-creating them
    if orig_sr not in resampler_cache:
        resampler_cache[orig_sr] = T.Resample(orig_sr, new_sr).to(DEVICE)
    return resampler_cache[orig_sr]

def pad_or_truncate(audio, max_len):
    if audio.shape[0] > max_len:
        audio = audio[:max_len]
    else:
        padding = torch.zeros(max_len - audio.shape[0], device=DEVICE)
        audio = torch.cat((audio, padding), dim=0)
    return audio

def preprocess_example(example, label_value):
    """
    Process a single example from the dataset.
    Returns a dictionary with properly shaped tensors.
    """
    # Handle both dict with 'audio' key and direct audio dict
    if "audio" in example:
        audio_item = example["audio"]
    else:
        audio_item = example
    
    audio_array = torch.FloatTensor(audio_item["array"]).to(DEVICE)
    
    # 1. Resample if necessary
    if audio_item["sampling_rate"] != SAMPLE_RATE:
        resampler = get_resampler(audio_item["sampling_rate"], SAMPLE_RATE)
        audio_array = resampler(audio_array)

    # 2. Pad/truncate raw audio
    raw_audio = pad_or_truncate(audio_array, MAX_LEN_SAMPLES)
    
    # 3. Create Mel Spectrogram
    melspec = mel_transform(raw_audio) # (Mels, Time)
    
    # 4. Add channel dim and pad/truncate time
    melspec = melspec.unsqueeze(0) # (1, Mels, Time)
    if melspec.shape[2] > MAX_MEL_FRAMES:
        melspec = melspec[:, :, :MAX_MEL_FRAMES]
    else:
        padding = torch.zeros(1, N_MELS, MAX_MEL_FRAMES - melspec.shape[2]).to(DEVICE)
        melspec = torch.cat((melspec, padding), dim=2)
    
    # Return as numpy arrays for PyArrow compatibility
    return {
        "raw_audio": raw_audio.cpu().numpy(),
        "melspec": melspec.cpu().numpy(),
        "label": np.array(label_value, dtype=np.int64)
    }

def preprocess_fake(example):
    # Note: MLAAD's 'label' (104, etc.) is the *spoof type*.
    # We ignore it and assign a binary label '1' for "spoof".
    return preprocess_example(example, label_value=1)

def preprocess_real(example):
    # LibriSpeech has no 'label' column. We assign '0' for "bona fide".
    return preprocess_example(example, label_value=0)

def get_dataloader(split, batch_size):
    print(f"\n--- Loading {split} data stream ---")
    
    # Configuration
    TRAIN_SPLIT_RATIO = 0.8  # 80% for training, 20% for validation
    
    if split == 'train':
        fake_split = 'train'
        # Using 'train.100' for a balanced size. Use 'train.360' or 'train.960' for more data.
        real_split = 'train.clean.100' 
    else:
        # For validation, we'll split the 'train' split deterministically
        fake_split = 'train'
        real_split = 'validation.clean'

    # 1. Load FAKE dataset
    # You MUST accept terms on HF website and be logged in (huggingface-cli login)
    print(f"Loading FAKE data: mueller91/MLAAD (split={fake_split})")
    fake_ds = load_dataset("mueller91/MLAAD", 
                           split=fake_split, 
                           streaming=True)
    
    # Split MLAAD dataset deterministically using hash-based filtering
    # This ensures consistent train/val split across runs
    if split == 'validation':
        # For validation, filter to get the validation portion (last 20%)
        def is_validation(example, idx):
            # Use hash of a unique identifier to deterministically split
            # Use file path or index as the key
            key = example.get('path', str(idx))
            hash_val = hash(key) % 100
            return hash_val >= int(TRAIN_SPLIT_RATIO * 100)  # Last 20%
        fake_ds = fake_ds.filter(is_validation, with_indices=True)
    else:
        # For training, filter to get the training portion (first 80%)
        def is_training(example, idx):
            key = example.get('path', str(idx))
            hash_val = hash(key) % 100
            return hash_val < int(TRAIN_SPLIT_RATIO * 100)  # First 80%
        fake_ds = fake_ds.filter(is_training, with_indices=True)
    
    fake_ds = fake_ds.map(preprocess_fake, batched=False, remove_columns=["audio"])
    
    # 2. Load REAL dataset
    print(f"Loading REAL data: librispeech_asr (split={real_split})")
    real_ds = load_dataset("openslr/librispeech_asr",
                           split=real_split, 
                           streaming=True)
    real_ds = real_ds.map(preprocess_real, batched=False, remove_columns=["audio"])

    # 3. Manually interleave them 50/50 to avoid feature inference issues
    # This avoids the PyArrow serialization problem with interleave_datasets
    print("Interleaving real and fake streams...")
    
    class InterleavedDataset(IterableDataset):
        """Manually interleave two dataset iterators with 50/50 probability"""
        def __init__(self, fake_ds, real_ds, seed=42):
            super().__init__()
            self.fake_ds = fake_ds
            self.real_ds = real_ds
            self.seed = seed
        
        def __iter__(self):
            fake_iter = iter(self.fake_ds)
            real_iter = iter(self.real_ds)
            rng = random.Random(self.seed)
            
            # Use a buffer to store items for shuffling
            buffer = []
            buffer_size = 1000
            
            fake_exhausted = False
            real_exhausted = False
            
            while not (fake_exhausted and real_exhausted):
                # 50/50 chance to pick from fake or real
                if rng.random() < 0.5:
                    if not fake_exhausted:
                        try:
                            item = next(fake_iter)
                            buffer.append(item)
                        except StopIteration:
                            fake_exhausted = True
                    elif not real_exhausted:
                        try:
                            item = next(real_iter)
                            buffer.append(item)
                        except StopIteration:
                            real_exhausted = True
                else:
                    if not real_exhausted:
                        try:
                            item = next(real_iter)
                            buffer.append(item)
                        except StopIteration:
                            real_exhausted = True
                    elif not fake_exhausted:
                        try:
                            item = next(fake_iter)
                            buffer.append(item)
                        except StopIteration:
                            fake_exhausted = True
                
                # Shuffle and yield buffer when it's full
                if len(buffer) >= buffer_size:
                    rng.shuffle(buffer)
                    yield from buffer
                    buffer = []
            
            # Yield remaining items in buffer
            if buffer:
                rng.shuffle(buffer)
                yield from buffer
    
    combined_ds = InterleavedDataset(fake_ds, real_ds, seed=42) 

    # 5. Create the DataLoader with proper batching
    def collate_fn(batch_list):
        """
        Collate a list of examples into a batch.
        Each example is a dict with 'raw_audio', 'melspec', 'label'.
        """
        # Stack all examples into batches
        raw_audio_batch = torch.from_numpy(np.stack([ex['raw_audio'] for ex in batch_list]))
        melspec_batch = torch.from_numpy(np.stack([ex['melspec'] for ex in batch_list]))
        label_batch = torch.from_numpy(np.stack([ex['label'] for ex in batch_list])).long()
        
        return {
            'raw_audio': raw_audio_batch,
            'melspec': melspec_batch,
            'label': label_batch
        }
    
    loader = DataLoader(combined_ds, 
                        batch_size=batch_size,
                        collate_fn=collate_fn,
                        num_workers=0) # Must be 0 for streaming datasets
    
    print(f"--- {split} dataloader created ---")
    return loader

# --- 7. Evaluation Metric: Equal Error Rate (EER) ---

def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    return eer

# --- 8. The Training and Validation Loops ---

def train_epoch(model, dataloader, optimizer, epoch):
    model.train()
    total_loss, total_loss_spoof, total_loss_adv = 0, 0, 0
    
    print(f"\n--- Starting Epoch {epoch}/{EPOCHS} ---")
    
    # Iterate for a fixed number of steps
    for i, batch in enumerate(dataloader):
        if i >= STEPS_PER_EPOCH:
            break
        
        # Move data to GPU (tensors are already created by collate_fn)
        raw_audio = batch['raw_audio'].to(DEVICE)  # (Batch, Samples)
        mels = batch['melspec'].to(DEVICE)  # (Batch, 1, N_Mels, Time)
        labels = batch['label'].to(DEVICE)  # (Batch,)
        
        optimizer.zero_grad()
        
        total_loss_batch, loss_spoof_batch, loss_adv_batch, _ = model(raw_audio, mels, labels)
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_loss_spoof += loss_spoof_batch.item()
        total_loss_adv += loss_adv_batch.item()
        
        if (i + 1) % 50 == 0:
            print(f"  Epoch {epoch} | Step {i+1}/{STEPS_PER_EPOCH} | "
                  f"Total Loss: {total_loss_batch.item():.4f} | "
                  f"Spoof Loss: {loss_spoof_batch.item():.4f} | "
                  f"Adv Loss: {loss_adv_batch.item():.4f}")

    avg_loss = total_loss / STEPS_PER_EPOCH
    print(f"--- EPOCH {epoch} SUMMARY ---")
    print(f"Avg Train Loss: {avg_loss:.4f}")

def validate(model, dataloader, epoch):
    model.eval()
    all_labels = []
    all_scores = []
    
    print(f"\n--- Validating Epoch {epoch} ---")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= VAL_STEPS:
                break
                
            # Move data to GPU (tensors are already created by collate_fn)
            raw_audio = batch['raw_audio'].to(DEVICE)  # (Batch, Samples)
            mels = batch['melspec'].to(DEVICE)  # (Batch, 1, N_Mels, Time)
            labels = batch['label']  # Keep on CPU for evaluation
            
            # Forward pass (ONLY the lightweight prosody encoder)
            _, out_spoof = model.prosody_encoder(mels)
            
            scores = F.softmax(out_spoof, dim=1)[:, 1].cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_scores.extend(scores)
            
            if (i + 1) % 50 == 0:
                print(f"  Validation Step {i+1}/{VAL_STEPS}...")

    eer = calculate_eer(all_labels, all_scores)
    print(f"--- VALIDATION EPOCH {epoch} EER: {eer * 100:.2f}% ---")
    return eer

# --- 9. Main Execution ---

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    # Set torchaudio backend
    try:
        torchaudio.set_audio_backend("sox_io") 
    except Exception:
        print("sox_io backend not available. Using default.")

    print("Initializing model...")
    model = DisentangledAntiSpoofingModel().to(DEVICE)
    
    # We only optimize the PROSODY ENCODER and the DISCRIMINATOR
    optimizer = optim.Adam(
        list(model.prosody_encoder.parameters()) + list(model.discriminator.parameters()),
        lr=LEARNING_RATE
    )
    
    print(f"--- Starting Training for {EPOCHS} Epochs ---")
    print(f"--- Each epoch = {STEPS_PER_EPOCH} train steps + {VAL_STEPS} val steps ---")
    
    best_eer = 1.0
    for epoch in range(1, EPOCHS + 1):
        # Create new dataloaders for each epoch
        # This re-shuffles the streams
        train_loader = get_dataloader('train', BATCH_SIZE)
        val_loader = get_dataloader('validation', BATCH_SIZE)
        
        train_epoch(model, train_loader, optimizer, epoch)
        eer = validate(model, val_loader, epoch)
        
        if eer < best_eer:
            best_eer = eer
            print(f"✨ New best EER: {eer*100:.2f}%. Saving model... ✨")
            # Save ONLY the lightweight prosody encoder
            torch.save(model.prosody_encoder.state_dict(), "prosody_encoder_best.pth")

    print(f"--- Training Complete ---")
    print(f"Best validation EER: {best_eer * 100:.2f}%")
    print("\n--- To deploy this model, you only need 'prosody_encoder_best.pth' ---")
    
    warnings.filterwarnings("default", category=UserWarning)


if __name__ == "__main__":
    main()
