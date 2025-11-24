import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.autograd import Function
import itertools
import random
import numpy as np
import warnings
import os

# For loading models and data
from transformers import Wav2Vec2Model
from datasets import load_dataset, Audio
import torchaudio.transforms as T
import torchaudio

# For evaluation
from sklearn.metrics import roc_curve

# ==========================================
# ⚙️ GLOBAL CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable CUDNN Benchmark for speed on Nvidia GPUs
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

print(f"--- Using device: {DEVICE} ---")

# --- Data Params ---
SAMPLE_RATE = 16000
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
N_MELS = 80
MAX_AUDIO_LEN_SECONDS = 4
MAX_LEN_SAMPLES = SAMPLE_RATE * MAX_AUDIO_LEN_SECONDS
# Calc frames: (4s * 16000) / 160 hop_length = 400 frames + 1
MAX_MEL_FRAMES = int(MAX_AUDIO_LEN_SECONDS * (SAMPLE_RATE / 160)) + 1

# --- Training Params ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50 
STEPS_PER_EPOCH = 500  # How many batches per epoch (since data is streaming)
VAL_STEPS = 100        # How many batches to validate
LAMBDA_ADV = 1.0       # Base adversarial weight (will be scaled dynamically)


# ==========================================
# 1. MODEL COMPONENTS
# ==========================================

# --- Gradient Reversal Layer (The "Disentanglement" Magic) ---
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient (multiply by -lambda)
        return (grad_output.neg() * ctx.lambda_), None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# --- Prosody Encoder (The "Student" / Detector) ---
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
        
        # ResNet-style blocks to capture temporal prosody
        self.layer1 = ResBlock(64, 64, stride=(1,2))
        self.layer2 = ResBlock(64, 128, stride=(2,2)) 
        self.layer3 = ResBlock(128, 256, stride=(2,2))
        self.layer4 = ResBlock(256, 512, stride=(2,2))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output two things: Features (for adversary) and Class (Real/Fake)
        self.fc1 = nn.Linear(512, num_features)
        self.fc2 = nn.Linear(num_features, 2) 

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x) 
        x = x.view(x.size(0), -1) 
        
        features = F.relu(self.fc1(x)) 
        out_spoof = self.fc2(features) 
        return features, out_spoof

# --- Content Encoder (The "Teacher" - Frozen) ---
class ContentEncoder(nn.Module):
    def __init__(self, model_name=WAV2VEC2_MODEL_NAME):
        super(ContentEncoder, self).__init__()
        print(f"Loading frozen {model_name}...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze it completely
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        self.feature_dim = self.wav2vec2.config.hidden_size 

    def forward(self, x):
        with torch.no_grad(): 
            outputs = self.wav2vec2(x, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state 
            # Mean pool to get a single content vector per audio file
            content_features = torch.mean(hidden_states, dim=1) 
        return content_features

# --- Content Discriminator (The "Adversary") ---
class ContentDiscriminator(nn.Module):
    def __init__(self, prosody_dim=256, content_dim=768, hidden_dim=256):
        super(ContentDiscriminator, self).__init__()
        # The GRL layer is what flips the game
        self.grl = GradientReversalLayer(lambda_=1.0) # Base lambda
        self.net = nn.Sequential(
            nn.Linear(prosody_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, content_dim) # Tries to guess the Wav2Vec2 features
        )
        self.loss_fn = nn.MSELoss() 

    def forward(self, prosody_features, content_features):
        # 1. Reverse gradients
        prosody_features_adv = self.grl(prosody_features)
        # 2. Try to predict content
        predicted_content = self.net(prosody_features_adv)
        # 3. Calculate error
        loss_adv = self.loss_fn(predicted_content, content_features)
        return loss_adv

# --- The Full Model Wrapper ---
class DisentangledAntiSpoofingModel(nn.Module):
    def __init__(self):
        super(DisentangledAntiSpoofingModel, self).__init__()
        self.content_encoder = ContentEncoder()
        self.prosody_encoder = ProsodyEncoder(num_features=256)
        self.discriminator = ContentDiscriminator(prosody_dim=256, 
                                                content_dim=self.content_encoder.feature_dim)
    
    def forward(self, raw_audio, mels, labels):
        # Kept for structure, but we usually call sub-modules manually in train_epoch
        # to control the lambda dynamic scaling.
        prosody_features, out_spoof = self.prosody_encoder(mels)
        return prosody_features, out_spoof


# ==========================================
# 2. DATA PROCESSING (CPU OPTIMIZED)
# ==========================================

# PERFORMANCE FIX 1: Keep Transform on CPU
# This allows multiple CPU workers to process audio in parallel
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=N_MELS
)

resampler_cache = {}

def get_resampler(orig_sr, new_sr):
    # Cache resamplers to avoid re-creating them
    if orig_sr not in resampler_cache:
        resampler_cache[orig_sr] = T.Resample(orig_sr, new_sr) # On CPU
    return resampler_cache[orig_sr]

def pad_or_truncate(audio, max_len):
    if audio.shape[0] > max_len:
        audio = audio[:max_len]
    else:
        padding = torch.zeros(max_len - audio.shape[0])
        audio = torch.cat((audio, padding), dim=0)
    return audio

def preprocess_example(example, label_value):
    """
    Process a single example from the dataset.
    Returns a dictionary with properly shaped tensors on CPU.
    """
    if "audio" in example:
        audio_item = example["audio"]
    else:
        audio_item = example
    
    # 1. Load audio (CPU)
    audio_array = torch.FloatTensor(audio_item["array"])
    
    # 2. Resample if necessary (CPU)
    if audio_item["sampling_rate"] != SAMPLE_RATE:
        resampler = get_resampler(audio_item["sampling_rate"], SAMPLE_RATE)
        audio_array = resampler(audio_array)

    # 3. Pad/truncate (CPU)
    raw_audio = pad_or_truncate(audio_array, MAX_LEN_SAMPLES)
    
    # 4. Create Mel Spectrogram (CPU)
    melspec = mel_transform(raw_audio)
    
    # 5. Add channel dim
    melspec = melspec.unsqueeze(0)
    if melspec.shape[2] > MAX_MEL_FRAMES:
        melspec = melspec[:, :, :MAX_MEL_FRAMES]
    else:
        padding = torch.zeros(1, N_MELS, MAX_MEL_FRAMES - melspec.shape[2])
        melspec = torch.cat((melspec, padding), dim=2)
    
    # Return as numpy arrays for efficient PyArrow serialization
    return {
        "raw_audio": raw_audio.numpy(),
        "melspec": melspec.numpy(),
        "label": np.array(label_value, dtype=np.int64)
    }

def preprocess_fake(example): return preprocess_example(example, 1)
def preprocess_real(example): return preprocess_example(example, 0)


# ==========================================
# 3. DATA LOADING (MULTI-WORKER SUPPORT)
# ==========================================

def get_dataloader(split, batch_size):
    print(f"\n--- Loading {split} data stream ---")
    
    TRAIN_SPLIT_RATIO = 0.8 
    
    if split == 'train':
        fake_split = 'train'
        real_split = 'train.clean.100' 
    else:
        fake_split = 'train'
        real_split = 'validation.clean'

    # Load Fake Data
    print(f"Loading FAKE data: mueller91/MLAAD ({fake_split})")
    fake_ds = load_dataset("mueller91/MLAAD", split=fake_split, streaming=True)
    
    # Deterministic Split for Validation
    if split == 'validation':
        fake_ds = fake_ds.filter(
            lambda x, idx: (hash(x.get('path', str(idx))) % 100) >= int(TRAIN_SPLIT_RATIO * 100), 
            with_indices=True
        )
    else:
        fake_ds = fake_ds.filter(
            lambda x, idx: (hash(x.get('path', str(idx))) % 100) < int(TRAIN_SPLIT_RATIO * 100), 
            with_indices=True
        )
    
    # Map with CPU processing
    fake_ds = fake_ds.map(preprocess_fake, batched=False, remove_columns=["audio"])
    
    # Load Real Data
    print(f"Loading REAL data: librispeech_asr ({real_split})")
    real_ds = load_dataset("openslr/librispeech_asr", split=real_split, streaming=True)
    real_ds = real_ds.map(preprocess_real, batched=False, remove_columns=["audio"])

    print("Interleaving real and fake streams...")
    
    # Custom Interleaver to handle Workers correctly
    class InterleavedDataset(IterableDataset):
        def __init__(self, fake_ds, real_ds, seed=42):
            super().__init__()
            self.fake_ds = fake_ds
            self.real_ds = real_ds
            self.seed = seed
        
        def __iter__(self):
            # PERFORMANCE FIX: Ensure workers have different seeds
            worker_info = get_worker_info()
            if worker_info is None:
                actual_seed = self.seed
            else:
                actual_seed = self.seed + worker_info.id
            
            fake_iter = iter(self.fake_ds)
            real_iter = iter(self.real_ds)
            rng = random.Random(actual_seed)
            
            buffer = []
            buffer_size = 1000
            
            # Infinite loop that breaks on StopIteration
            while True:
                try:
                    # 50/50 chance
                    if rng.random() < 0.5:
                        buffer.append(next(fake_iter))
                    else:
                        buffer.append(next(real_iter))
                except StopIteration:
                    break # End of stream
                
                if len(buffer) >= buffer_size:
                    rng.shuffle(buffer)
                    yield from buffer
                    buffer = []
            
            # Yield remaining
            if buffer:
                rng.shuffle(buffer)
                yield from buffer
    
    combined_ds = InterleavedDataset(fake_ds, real_ds, seed=42) 

    # Collate Fn to stack CPU tensors into Batches
    def collate_fn(batch_list):
        raw_audio_batch = torch.from_numpy(np.stack([ex['raw_audio'] for ex in batch_list]))
        melspec_batch = torch.from_numpy(np.stack([ex['melspec'] for ex in batch_list]))
        label_batch = torch.from_numpy(np.stack([ex['label'] for ex in batch_list])).long()
        
        return {
            'raw_audio': raw_audio_batch,
            'melspec': melspec_batch,
            'label': label_batch
        }
    
    # PERFORMANCE FIX: num_workers=4 and pin_memory=True
    loader = DataLoader(combined_ds, 
                        batch_size=batch_size,
                        collate_fn=collate_fn,
                        num_workers=4, # Use 4 CPU cores
                        pin_memory=True if torch.cuda.is_available() else False) 
    
    print(f"--- {split} dataloader created ---")
    return loader


# ==========================================
# 4. TRAINING UTILS (EVAL & LOOP)
# ==========================================

def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    return eer

def train_epoch(model, dataloader, optimizer, epoch):
    model.train()
    total_loss_accum, total_loss_spoof, total_loss_adv = 0, 0, 0
    
    # --- SOTA TRICK 1: Adversarial Warmup ---
    # Epoch 1-5: Lambda = 0.0 (Pure Classification) -> Learn basic features
    # Epoch 6+:  Lambda = 1.0 (Disentanglement) -> Force robustness
    current_lambda = 0.0 if epoch <= 5 else 1.0
    
    print(f"\n--- Starting Epoch {epoch}/{EPOCHS} (Lambda_Adv: {current_lambda}) ---")
    
    for i, batch in enumerate(dataloader):
        if i >= STEPS_PER_EPOCH:
            break
        
        # Move to GPU (non_blocking=True for overlap with CPU load)
        raw_audio = batch['raw_audio'].to(DEVICE, non_blocking=True)
        mels = batch['melspec'].to(DEVICE, non_blocking=True)
        labels = batch['label'].to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 1. Main Forward Pass
        prosody_features, out_spoof = model.prosody_encoder(mels)
        loss_spoof = F.cross_entropy(out_spoof, labels)
        
        # 2. Adversarial Forward Pass
        content_features = model.content_encoder(raw_audio)
        loss_adv = model.discriminator(prosody_features, content_features.detach())
        
        # --- SOTA FIX: Force Scaling ---
        # Spoof Loss is ~0.2. Adv Loss is ~0.01. 
        # Multiplying by 10.0 brings Adv Loss to ~0.1, making it impactful.
        # current_lambda handles the warmup.
        total_loss_batch = loss_spoof + (loss_adv * 10.0 * current_lambda)
        
        total_loss_batch.backward()
        
        # --- SOTA TRICK 2: Gradient Clipping ---
        # Prevents "Exploding Gradients" that cause the 50% EER crash
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        
        optimizer.step()
        
        total_loss_accum += total_loss_batch.item()
        total_loss_spoof += loss_spoof.item()
        total_loss_adv += loss_adv.item()
        
        if (i + 1) % 50 == 0:
            print(f"  Epoch {epoch} | Step {i+1}/{STEPS_PER_EPOCH} | "
                  f"Total: {total_loss_batch.item():.4f} | Spoof: {loss_spoof.item():.4f} | Adv: {loss_adv.item():.4f}")

    avg_loss = total_loss_accum / STEPS_PER_EPOCH
    print(f"--- EPOCH {epoch} SUMMARY --- Avg Train Loss: {avg_loss:.4f}")

def validate(model, dataloader, epoch):
    model.eval()
    all_labels = []
    all_scores = []
    
    print(f"\n--- Validating Epoch {epoch} ---")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= VAL_STEPS:
                break
                
            mels = batch['melspec'].to(DEVICE, non_blocking=True)
            labels = batch['label'] # Keep on CPU
            
            # Forward pass (only the part we deploy)
            _, out_spoof = model.prosody_encoder(mels)
            
            # Score: Probability of being Fake (Class 1)
            scores = F.softmax(out_spoof, dim=1)[:, 1].cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_scores.extend(scores)
            
            if (i + 1) % 50 == 0:
                print(f"  Validation Step {i+1}/{VAL_STEPS}...")

    eer = calculate_eer(all_labels, all_scores)
    print(f"--- VALIDATION EPOCH {epoch} EER: {eer * 100:.2f}% ---")
    return eer


# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Fix for some torchaudio backends
    try:
        torchaudio.set_audio_backend("sox_io") 
    except Exception:
        print("sox_io backend not available. Using default.")

    print("Initializing model...")
    model = DisentangledAntiSpoofingModel().to(DEVICE)
    
    optimizer = optim.Adam(
        list(model.prosody_encoder.parameters()) + list(model.discriminator.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-5 # SOTA Regularization
    )
    
    # --- SOTA TRICK 3: Learning Rate Scheduler ---
    # If EER stops improving, lower the learning rate to fine-tune.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    print(f"--- Starting Training for {EPOCHS} Epochs ---")
    
    best_eer = 1.0
    for epoch in range(1, EPOCHS + 1):
        # Re-create dataloaders to shuffle the infinite stream
        train_loader = get_dataloader('train', BATCH_SIZE)
        val_loader = get_dataloader('validation', BATCH_SIZE)
        
        train_epoch(model, train_loader, optimizer, epoch)
        eer = validate(model, val_loader, epoch)
        
        # Update LR
        scheduler.step(eer)
        
        if eer < best_eer:
            best_eer = eer
            print(f"✨ New best EER: {eer*100:.2f}%. Saving model... ✨")
            torch.save(model.prosody_encoder.state_dict(), "prosody_encoder_best.pth")

    print(f"--- Training Complete ---")
    print(f"Best validation EER: {best_eer * 100:.2f}%")
    print("\n--- To deploy this model, you only need 'prosody_encoder_best.pth' ---")
    
    warnings.filterwarnings("default", category=UserWarning)


if __name__ == "__main__":
    main()