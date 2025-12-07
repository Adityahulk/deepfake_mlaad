"""
AASIST Model Test Script
========================
Evaluates the AASIST-inspired model on ASVspoof 2019 LA dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import os
import math

# ==========================================
# CONFIG
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
MAX_AUDIO_LEN_SECONDS = 4
MAX_LEN_SAMPLES = SAMPLE_RATE * MAX_AUDIO_LEN_SECONDS

# Model params (must match train_v2.py)
SINC_OUT_CHANNELS = 70
ENCODER_DIM = 128


# ==========================================
# MODEL DEFINITION (Copy from train_v2.py)
# ==========================================

class SincConv(nn.Module):
    """Sinc-based learnable filterbank"""
    def __init__(self, out_channels=70, kernel_size=251, sample_rate=16000, 
                 min_low_hz=50, min_band_hz=50):
        super().__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
        mel_low = 2595 * np.log10(1 + low_hz / 700)
        mel_high = 2595 * np.log10(1 + high_hz / 700)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        self.low_hz_ = nn.Parameter(torch.Tensor(hz_points[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_points)).view(-1, 1))
        
        n_lin = torch.linspace(0, (kernel_size / 2) - 1, steps=kernel_size // 2)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)
        
        n = (kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate
        
    def forward(self, x):
        self.n_ = self.n_.to(x.device)
        self.window_ = self.window_.to(x.device)
        
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), 
                          self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]
        
        f_low = low / self.sample_rate
        f_high = high / self.sample_rate
        
        band_pass_left = ((torch.sin(f_high * self.n_) - torch.sin(f_low * self.n_)) / 
                         (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        if x.dim() == 3:
            b, c, t = x.size()
            y = x.mean(dim=2)
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
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.out = nn.Linear(out_dim, out_dim)
        
    def forward(self, x):
        b, n, _ = x.size()
        
        q = self.q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        
        return self.out(out)


class AASISTEncoder(nn.Module):
    def __init__(self, sinc_channels=70, encoder_dim=128):
        super().__init__()
        
        self.sinc = SincConv(out_channels=sinc_channels, kernel_size=251)
        self.sinc_bn = nn.BatchNorm1d(sinc_channels)
        self.sinc_pool = nn.MaxPool1d(3)
        
        self.res2net = nn.Sequential(
            Res2NetBlock(sinc_channels, 64, scale=4),
            nn.MaxPool1d(3),
            Res2NetBlock(64, 128, scale=4),
            nn.MaxPool1d(3),
            Res2NetBlock(128, 256, scale=4),
            nn.MaxPool1d(3),
            Res2NetBlock(256, encoder_dim, scale=4),
        )
        
        self.gat = GraphAttentionLayer(encoder_dim, encoder_dim, num_heads=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(encoder_dim, 2)
        
    def forward(self, x):
        x = self.sinc(x)
        x = F.relu(self.sinc_bn(x))
        x = self.sinc_pool(x)
        x = self.res2net(x)
        x = x.transpose(1, 2)
        x = x + self.gat(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return x, logits


# ==========================================
# DATASET
# ==========================================

class ASVspoof2019Dataset(Dataset):
    """ASVspoof 2019 LA dataset for evaluation"""
    
    def __init__(self, base_dir, split='eval'):
        self.base_dir = base_dir
        
        if split == 'eval':
            self.flac_dir = os.path.join(base_dir, 'ASVspoof2019_LA_eval/flac')
            protocol_path = os.path.join(base_dir, 
                'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
        else:  # dev
            self.flac_dir = os.path.join(base_dir, 'ASVspoof2019_LA_dev/flac')
            protocol_path = os.path.join(base_dir, 
                'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt')
        
        print(f"Reading protocol from: {protocol_path}")
        self.labels = pd.read_csv(protocol_path, sep=" ", header=None, 
                                  names=["spk", "filename", "sys", "unused", "key"])
        self.labels['target'] = self.labels['key'].apply(lambda x: 0 if x == 'bonafide' else 1)
        
        print(f"Loaded {len(self.labels)} samples")
        print(f"  Bonafide: {(self.labels['target'] == 0).sum()}")
        print(f"  Spoof: {(self.labels['target'] == 1).sum()}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        filename = row['filename']
        target = row['target']
        
        filepath = os.path.join(self.flac_dir, filename + ".flac")
        waveform, sr = torchaudio.load(filepath)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad/truncate
        if waveform.shape[1] > MAX_LEN_SAMPLES:
            waveform = waveform[:, :MAX_LEN_SAMPLES]
        else:
            padding = torch.zeros(1, MAX_LEN_SAMPLES - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        
        return waveform, target


# ==========================================
# EVALUATION
# ==========================================

def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[eer_index] + fnr[eer_index]) / 2


def evaluate(model_path, data_dir, split='eval'):
    print(f"\n{'='*60}")
    print(f"Evaluating AASIST Model on ASVspoof 2019 LA {split}")
    print(f"{'='*60}")
    
    # Load model
    model = AASISTEncoder(sinc_channels=SINC_OUT_CHANNELS, 
                          encoder_dim=ENCODER_DIM).to(DEVICE)
    
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Loaded model from {model_path}")
    
    # Load dataset
    dataset = ASVspoof2019Dataset(data_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Evaluate
    all_labels = []
    all_scores = []
    
    print(f"\nProcessing {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, (waveforms, labels) in enumerate(dataloader):
            waveforms = waveforms.to(DEVICE)
            
            _, logits = model(waveforms)
            probs = F.softmax(logits, dim=1)[:, 1]  # P(spoof)
            
            all_labels.extend(labels.numpy())
            all_scores.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {(batch_idx + 1) * 32} / {len(dataset)}...")
    
    # Calculate EER
    eer = calculate_eer(all_labels, all_scores)
    
    print(f"\n{'='*60}")
    print(f"🎯 ASVspoof 2019 LA {split.upper()} EER: {eer * 100:.2f}%")
    print(f"{'='*60}")
    
    return eer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='aasist_best.pth',
                        help='Path to model weights')
    parser.add_argument('--data_dir', type=str, default='./LA',
                        help='Path to ASVspoof LA directory')
    parser.add_argument('--split', type=str, default='eval', choices=['eval', 'dev'],
                        help='Dataset split to evaluate')
    
    args = parser.parse_args()
    
    if os.path.exists(args.model_path) and os.path.exists(args.data_dir):
        evaluate(args.model_path, args.data_dir, args.split)
    else:
        if not os.path.exists(args.model_path):
            print(f"❌ Model not found: {args.model_path}")
        if not os.path.exists(args.data_dir):
            print(f"❌ Data directory not found: {args.data_dir}")
