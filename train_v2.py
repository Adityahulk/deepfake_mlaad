"""
SOTA Anti-Spoofing Training Script v2
=====================================
AASIST-inspired architecture with RawBoost augmentation.
Training: MLAAD + LibriSpeech
Target: Cross-dataset generalization to ASVspoof 2019 (<15% EER)

Key improvements over v1:
1. AASIST-like encoder (Res2Net + SE + Graph Attention)
2. RawBoost augmentation for domain invariance
3. Focal Loss for hard example mining
4. OneCycleLR scheduler
5. Mixed precision training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
import warnings
import os

from transformers import Wav2Vec2Model
from datasets import load_dataset, Audio, interleave_datasets
import torchaudio.transforms as T
from sklearn.metrics import roc_curve

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

print(f"--- Using device: {DEVICE} ---")

# Audio params
SAMPLE_RATE = 16000
MAX_AUDIO_LEN_SECONDS = 4
MAX_LEN_SAMPLES = SAMPLE_RATE * MAX_AUDIO_LEN_SECONDS

# Training params
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
STEPS_PER_EPOCH = 500
VAL_STEPS = 100

# Model params
SINC_OUT_CHANNELS = 70
RES2NET_SCALE = 4  # Res2Net scale factor
ENCODER_DIM = 128
NUM_GAT_HEADS = 4


# ==========================================
# 1. RAWBOOST AUGMENTATION
# ==========================================
class RawBoost:
    """
    RawBoost augmentation for domain-invariant anti-spoofing.
    Reference: https://arxiv.org/abs/2111.04433
    """
    
    @staticmethod
    def linear_frequency_mod(x, fs=16000):
        """Linear Frequency Modulation (LFM) - simulates channel effects"""
        n_samples = len(x)
        # Random frequency sweep parameters
        f0 = random.uniform(0.1, 8) * 1000  # Start freq
        f1 = random.uniform(0.1, 8) * 1000  # End freq
        t = np.arange(n_samples) / fs
        
        # Phase modulation
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / (n_samples / fs))
        mod_signal = np.cos(phase) * random.uniform(0.001, 0.01)
        
        return x + mod_signal.astype(np.float32)
    
    @staticmethod
    def impulsive_signal(x, fs=16000):
        """Impulsive Signal Injection (ISI) - simulates clicks/pops"""
        n_samples = len(x)
        n_impulses = random.randint(1, int(n_samples / 1000))
        
        impulse_positions = np.random.randint(0, n_samples, n_impulses)
        impulse_values = np.random.uniform(-0.05, 0.05, n_impulses)
        
        x_aug = x.copy()
        for pos, val in zip(impulse_positions, impulse_values):
            x_aug[pos] = np.clip(x_aug[pos] + val, -1, 1)
        
        return x_aug
    
    @staticmethod
    def stationary_signal(x, fs=16000):
        """Stationary Signal Insertion (SSI) - simulates background noise"""
        n_samples = len(x)
        
        # Generate colored noise
        noise_type = random.choice(['white', 'pink', 'brown'])
        
        if noise_type == 'white':
            noise = np.random.randn(n_samples)
        elif noise_type == 'pink':
            # Pink noise via filtering
            noise = np.random.randn(n_samples)
            b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
            a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
            from scipy.signal import lfilter
            noise = lfilter(b, a, noise)
        else:  # brown
            noise = np.cumsum(np.random.randn(n_samples))
            noise = noise / np.max(np.abs(noise))
        
        snr = random.uniform(20, 40)  # dB
        noise_level = np.sqrt(np.mean(x ** 2)) / (10 ** (snr / 20))
        
        return x + (noise * noise_level).astype(np.float32)
    
    @staticmethod
    def augment(x, algo=None):
        """Apply random RawBoost augmentation"""
        if algo is None:
            algo = random.choice([0, 1, 2, 3, 4, 5])
        
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        
        if algo == 0:  # No augmentation
            return x
        elif algo == 1:  # LFM only
            return RawBoost.linear_frequency_mod(x)
        elif algo == 2:  # ISI only
            return RawBoost.impulsive_signal(x)
        elif algo == 3:  # SSI only
            try:
                return RawBoost.stationary_signal(x)
            except:
                return x
        elif algo == 4:  # LFM + ISI
            x = RawBoost.linear_frequency_mod(x)
            return RawBoost.impulsive_signal(x)
        else:  # All three
            x = RawBoost.linear_frequency_mod(x)
            x = RawBoost.impulsive_signal(x)
            try:
                x = RawBoost.stationary_signal(x)
            except:
                pass
            return x


# ==========================================
# 2. AASIST-INSPIRED MODEL COMPONENTS
# ==========================================

class SincConv(nn.Module):
    """
    Sinc-based learnable filterbank.
    Learns optimal bandpass filters directly from raw audio.
    """
    def __init__(self, out_channels=70, kernel_size=251, sample_rate=16000, 
                 min_low_hz=50, min_band_hz=50):
        super().__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # Initialize filterbank with mel-scale frequencies
        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
        mel_low = 2595 * np.log10(1 + low_hz / 700)
        mel_high = 2595 * np.log10(1 + high_hz / 700)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        # Learnable low and band frequencies
        self.low_hz_ = nn.Parameter(torch.Tensor(hz_points[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_points)).view(-1, 1))
        
        # Hamming window
        n_lin = torch.linspace(0, (kernel_size / 2) - 1, steps=kernel_size // 2)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)
        
        n = (kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate
        
    def forward(self, x):
        """x: (batch, 1, time)"""
        self.n_ = self.n_.to(x.device)
        self.window_ = self.window_.to(x.device)
        
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), 
                          self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]
        
        f_low = low / self.sample_rate
        f_high = high / self.sample_rate
        
        # Sinc filters
        band_pass_left = ((torch.sin(f_high * self.n_) - torch.sin(f_low * self.n_)) / 
                         (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        # x: (B, C, T) or (B, C, H, W)
        if x.dim() == 3:
            b, c, t = x.size()
            y = x.mean(dim=2)  # Global average pooling
        else:
            b, c, h, w = x.size()
            y = x.mean(dim=[2, 3])
            
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        
        if x.dim() == 3:
            return x * y.unsqueeze(2)
        else:
            return x * y.unsqueeze(2).unsqueeze(3)


class Res2NetBlock(nn.Module):
    """
    Res2Net block with multi-scale feature extraction.
    Better than standard ResNet for detecting fine-grained artifacts.
    """
    def __init__(self, in_channels, out_channels, scale=4, stride=1):
        super().__init__()
        
        width = out_channels // scale
        self.scale = scale
        self.width = width
        
        self.conv1 = nn.Conv1d(in_channels, width * scale, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, 3, stride=stride, padding=1, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])
        
        self.conv3 = nn.Conv1d(width * scale, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Split into groups
        spx = torch.split(out, self.width, dim=1)
        
        sp = []
        for i in range(self.scale):
            if i == 0:
                sp.append(spx[i])
            elif i == 1:
                sp.append(F.relu(self.bns[i-1](self.convs[i-1](spx[i]))))
            else:
                sp.append(F.relu(self.bns[i-1](self.convs[i-1](spx[i] + sp[-1]))))
        
        out = torch.cat(sp, dim=1)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        
        out += self.shortcut(identity)
        return F.relu(out)


class GraphAttentionLayer(nn.Module):
    """Graph Attention for spectro-temporal modeling"""
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.out = nn.Linear(out_dim, out_dim)
        
    def forward(self, x):
        """x: (batch, nodes, features)"""
        b, n, _ = x.size()
        
        q = self.q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        
        return self.out(out)


class AASISTEncoder(nn.Module):
    """
    AASIST-inspired encoder for anti-spoofing.
    Combines:
    1. Sinc filterbank front-end
    2. Res2Net blocks with SE attention
    3. Graph attention for spectro-temporal modeling
    """
    def __init__(self, sinc_channels=70, encoder_dim=128):
        super().__init__()
        
        # Front-end: Sinc filters
        self.sinc = SincConv(out_channels=sinc_channels, kernel_size=251)
        self.sinc_bn = nn.BatchNorm1d(sinc_channels)
        self.sinc_pool = nn.MaxPool1d(3)
        
        # Encoder: Res2Net blocks
        self.res2net = nn.Sequential(
            Res2NetBlock(sinc_channels, 64, scale=4),
            nn.MaxPool1d(3),
            Res2NetBlock(64, 128, scale=4),
            nn.MaxPool1d(3),
            Res2NetBlock(128, 256, scale=4),
            nn.MaxPool1d(3),
            Res2NetBlock(256, encoder_dim, scale=4),
        )
        
        # Graph attention for spectro-temporal modeling
        self.gat = GraphAttentionLayer(encoder_dim, encoder_dim, num_heads=4)
        
        # Output
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(encoder_dim, 2)
        
    def forward(self, x):
        """
        x: (batch, 1, time) - raw waveform
        Returns: logits (batch, 2)
        """
        # Sinc filterbank
        x = self.sinc(x)  # (B, sinc_channels, T)
        x = F.relu(self.sinc_bn(x))
        x = self.sinc_pool(x)
        
        # Res2Net encoder
        x = self.res2net(x)  # (B, encoder_dim, T')
        
        # Graph attention (treat time steps as nodes)
        x = x.transpose(1, 2)  # (B, T', encoder_dim)
        x = x + self.gat(x)  # Residual connection
        x = x.transpose(1, 2)  # (B, encoder_dim, T')
        
        # Pool and classify
        x = self.pool(x).squeeze(-1)  # (B, encoder_dim)
        logits = self.fc(x)  # (B, 2)
        
        return x, logits  # Return features for potential adversarial training


# ==========================================
# 3. LOSSES
# ==========================================

class FocalLoss(nn.Module):
    """Focal Loss for hard example mining"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ==========================================
# 4. DATA LOADING
# ==========================================

class StreamingAudioDataset(IterableDataset):
    """Streaming dataset for MLAAD + LibriSpeech with RawBoost"""
    
    def __init__(self, split='train', use_augmentation=True):
        super().__init__()
        self.split = split
        self.use_augmentation = use_augmentation
        
    def _get_streams(self):
        # Load fake data (MLAAD)
        print(f"Loading FAKE data: mueller91/MLAAD ({self.split})")
        fake_ds = load_dataset(
            "mueller91/MLAAD", 
            split="train",  # MLAAD only has train split
            streaming=True
        ).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
        
        # Load real data (LibriSpeech)
        real_split = "train.clean.100" if self.split == 'train' else "validation.clean"
        print(f"Loading REAL data: librispeech_asr ({real_split})")
        real_ds = load_dataset(
            "librispeech_asr", 
            split=real_split,
            streaming=True
        ).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
        
        return fake_ds, real_ds
    
    def __iter__(self):
        fake_ds, real_ds = self._get_streams()
        
        fake_iter = iter(fake_ds)
        real_iter = iter(real_ds)
        
        while True:
            try:
                # Alternate between fake and real
                if random.random() < 0.5:
                    item = next(fake_iter)
                    label = 1  # Fake/Spoof
                else:
                    item = next(real_iter)
                    label = 0  # Real/Bonafide
                    
                audio = item['audio']['array']
                
                # Apply augmentation during training
                if self.use_augmentation and self.split == 'train':
                    audio = RawBoost.augment(audio)
                
                # Convert to tensor
                audio = torch.FloatTensor(audio)
                
                # Pad/truncate
                if audio.shape[0] > MAX_LEN_SAMPLES:
                    audio = audio[:MAX_LEN_SAMPLES]
                else:
                    padding = torch.zeros(MAX_LEN_SAMPLES - audio.shape[0])
                    audio = torch.cat([audio, padding])
                
                yield audio.unsqueeze(0), label  # (1, T), label
                
            except StopIteration:
                # Restart iterators
                fake_ds, real_ds = self._get_streams()
                fake_iter = iter(fake_ds)
                real_iter = iter(real_ds)


def get_dataloader(split='train', batch_size=16):
    """Create dataloader for training or validation"""
    print(f"\n--- Loading {split} data stream ---")
    use_aug = (split == 'train')
    dataset = StreamingAudioDataset(split=split, use_augmentation=use_aug)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Streaming doesn't work well with multiple workers
        pin_memory=True
    )
    print(f"--- {split} dataloader created ---")
    return loader


# ==========================================
# 5. TRAINING FUNCTIONS
# ==========================================

def calculate_eer(labels, scores):
    """Calculate Equal Error Rate"""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[eer_index] + fnr[eer_index]) / 2


def train_epoch(model, dataloader, optimizer, scheduler, criterion, 
                scaler, epoch, steps_per_epoch):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    
    for step, (audio, labels) in enumerate(dataloader):
        if step >= steps_per_epoch:
            break
            
        audio = audio.to(DEVICE)  # (B, 1, T)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            _, logits = model(audio)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        
        if (step + 1) % 50 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{steps_per_epoch} | "
                  f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    return total_loss / min(step + 1, steps_per_epoch)


@torch.no_grad()
def validate(model, dataloader, val_steps):
    """Validate and compute EER"""
    model.eval()
    
    all_labels = []
    all_scores = []
    
    for step, (audio, labels) in enumerate(dataloader):
        if step >= val_steps:
            break
            
        audio = audio.to(DEVICE)
        
        _, logits = model(audio)
        probs = F.softmax(logits, dim=1)[:, 1]  # P(fake)
        
        all_labels.extend(labels.numpy())
        all_scores.extend(probs.cpu().numpy())
        
        if (step + 1) % 50 == 0:
            print(f"  Validation Step {step+1}/{val_steps}...")
    
    eer = calculate_eer(all_labels, all_scores)
    return eer


# ==========================================
# 6. MAIN TRAINING LOOP
# ==========================================

def main():
    print("\n" + "="*60)
    print("AASIST-Inspired Anti-Spoofing Training v2")
    print("="*60)
    
    # Initialize model
    print("\nInitializing AASIST encoder...")
    model = AASISTEncoder(sinc_channels=SINC_OUT_CHANNELS, 
                          encoder_dim=ENCODER_DIM).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                            weight_decay=WEIGHT_DECAY)
    
    # OneCycleLR scheduler
    total_steps = EPOCHS * STEPS_PER_EPOCH
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE * 10,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Track best model
    best_eer = float('inf')
    
    print(f"\n--- Starting Training for {EPOCHS} Epochs ---")
    
    for epoch in range(1, EPOCHS + 1):
        # Create fresh dataloaders each epoch (for streaming)
        train_loader = get_dataloader('train', BATCH_SIZE)
        val_loader = get_dataloader('validation', BATCH_SIZE)
        
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        # Train
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            scaler, epoch, STEPS_PER_EPOCH
        )
        print(f"--- EPOCH {epoch} SUMMARY --- Avg Loss: {avg_loss:.4f}")
        
        # Validate
        print(f"\n--- Validating Epoch {epoch} ---")
        eer = validate(model, val_loader, VAL_STEPS)
        print(f"--- VALIDATION EPOCH {epoch} EER: {eer*100:.2f}% ---")
        
        # Save best model
        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), "aasist_best.pth")
            print(f"✨ New best EER: {eer*100:.2f}%. Saved to aasist_best.pth ✨")
        
        # Early stopping check
        if eer < 0.10:  # <10% EER
            print(f"\n🎉 Achieved target EER < 10%! Stopping early.")
            break
    
    print("\n" + "="*60)
    print(f"Training Complete! Best EER: {best_eer*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
