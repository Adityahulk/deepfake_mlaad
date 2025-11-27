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

# --- Configuration (MUST MATCH TRAIN.PY) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
N_MELS = 80
MAX_AUDIO_LEN_SECONDS = 4
MAX_MEL_FRAMES = int(MAX_AUDIO_LEN_SECONDS * (SAMPLE_RATE / 160)) + 1

# --- 1. Model Definition (Copy of ProsodyEncoder from train.py) ---
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
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResBlock(64, 64, stride=(1,2))
        self.layer2 = ResBlock(64, 128, stride=(2,2)) 
        self.layer3 = ResBlock(128, 256, stride=(2,2))
        self.layer4 = ResBlock(256, 512, stride=(2,2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
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

# --- 2. ASVspoof Dataset Loader ---
class ASVspoof2019Dataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.flac_dir = os.path.join(base_dir, 'ASVspoof2019_LA_eval/flac')
        protocol_path = os.path.join(base_dir, 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
        
        print(f"Reading protocol from: {protocol_path}")
        # Columns: SPEAKER_ID, AUDIO_FILE_NAME, SYSTEM_ID, KEY (bonafide/spoof)
        self.labels = pd.read_csv(protocol_path, sep=" ", header=None, names=["spk", "filename", "sys", "unused", "key"])
        
        # Filter map: bonafide -> 0, spoof -> 1
        self.labels['target'] = self.labels['key'].apply(lambda x: 0 if x == 'bonafide' else 1)
        
        # Transform matches training exactly
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, win_length=400, hop_length=160, n_mels=N_MELS
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        filename = row['filename']
        target = row['target']
        
        filepath = os.path.join(self.flac_dir, filename + ".flac")
        
        waveform, sr = torchaudio.load(filepath)
        
        # 1. Resample
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # 2. Pad/Truncate
        max_len = SAMPLE_RATE * MAX_AUDIO_LEN_SECONDS
        if waveform.shape[1] > max_len:
            waveform = waveform[:, :max_len]
        else:
            padding = torch.zeros(1, max_len - waveform.shape[1])
            waveform = torch.cat((waveform, padding), dim=1)
            
        # 3. Mel Spectrogram
        melspec = self.mel_transform(waveform)
        melspec = melspec.unsqueeze(0) # (1, Mels, Time)
        
        # 4. Pad/Truncate Mels
        if melspec.shape[2] > MAX_MEL_FRAMES:
            melspec = melspec[:, :, :MAX_MEL_FRAMES]
        else:
            padding = torch.zeros(1, N_MELS, MAX_MEL_FRAMES - melspec.shape[2])
            melspec = torch.cat((melspec, padding), dim=2)
            
        return melspec, target

# --- 3. Evaluation Function ---
def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[eer_index] + fnr[eer_index]) / 2

def evaluate_model(model_path, data_dir):
    print(f"\n--- Evaluating Model: {model_path} ---")
    
    model = ProsodyEncoder().to(DEVICE)
    
    try:
        # Load weights (Strict=False allows ignoring keys if needed, but here structure should match)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()
    
    dataset = ASVspoof2019Dataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    all_scores = []
    all_labels = []
    
    print(f"Processing {len(dataset)} test files...")
    
    with torch.no_grad():
        for mels, labels in dataloader:
            mels = mels.to(DEVICE)
            
            # Forward pass
            _, out_spoof = model(mels)
            
            # Score: Probability of being Fake (Class 1)
            probs = F.softmax(out_spoof, dim=1)[:, 1]
            
            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    eer = calculate_eer(all_labels, all_scores)
    print(f"\n------------------------------------------------")
    print(f"RESULTS FOR {model_path}")
    print(f"ASVspoof 2019 LA Eval EER: {eer * 100:.2f}%")
    print(f"------------------------------------------------\n")

# --- 4. Main Execution ---
if __name__ == "__main__":
    DATA_DIR = "./LA"  # Ensure this points to your unzipped LA folder
    MODEL_FILE = "prosody_encoder_best.pth"
    
    if os.path.exists(MODEL_FILE) and os.path.exists(DATA_DIR):
        evaluate_model(MODEL_FILE, DATA_DIR)
    else:
        print("Error: Could not find model file or LA dataset directory.")