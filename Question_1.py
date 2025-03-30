#************ Task 1**********************

######################### For Part II #########################################

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import subprocess
import tempfile
from omegaconf import OmegaConf
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model
import gc
import psutil
import soundfile as sf
import io

# --------------------------
# 1. Configuration for Optimization
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(os.cpu_count()) if device.type == 'cpu' else None
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

# --------------------------
# 2. Memory-Efficient Audio Loading
# --------------------------
def load_audio(path, sample_rate=16000):
    """Optimized audio loading that minimizes disk usage"""
    try:
        # First try direct loading for supported formats
        if path.lower().endswith(('.wav', '.flac')):
            audio, _ = sf.read(path, dtype='float32', always_2d=False)
        else:
            # Use in-memory conversion for unsupported formats
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-ac', '1', '-ar', str(sample_rate),
                '-f', 'wav', '-'
            ]
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL
            )
            stdout, _ = process.communicate()

            # Read directly from memory
            audio, _ = sf.read(io.BytesIO(stdout), dtype='float32')

        if len(audio) == 0 or np.max(np.abs(audio)) < 0.001:
            return np.zeros(sample_rate * 3, dtype=np.float32)

        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return np.zeros(sample_rate * 3, dtype=np.float32)

# --------------------------
# 3. Optimized Dataset Class
# --------------------------
class VoxCelebDataset(Dataset):
    def __init__(self, root_dir, ids, sample_rate=16000, duration=3):
        self.root_dir = root_dir
        self.ids = ids
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * duration
        self.speaker_to_idx = {speaker: idx for idx, speaker in enumerate(ids)}
        self.samples = []

        # Build file list more efficiently
        for speaker in ids:
            speaker_dir = os.path.join(root_dir, speaker)
            if not os.path.exists(speaker_dir):
                continue

            for root, _, files in os.walk(speaker_dir):
                for file in files:
                    if file.endswith(('.wav', '.flac', '.m4a')):
                        full_path = os.path.join(root, file)
                        self.samples.append((full_path, self.speaker_to_idx[speaker]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        audio = load_audio(path, self.sample_rate)

        # More efficient padding/trimming
        if len(audio) > self.max_samples:
            start = np.random.randint(0, len(audio) - self.max_samples)
            audio = audio[start:start+self.max_samples]
        elif len(audio) < self.max_samples:
            padding = self.max_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')

        return torch.from_numpy(audio), label

# --------------------------
# 4. Model Loading (Optimized)
# --------------------------
def load_fairseq_model(checkpoint_path):
    """Load model with memory optimizations"""
    # Load with mmap for large files
    checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True)
    cfg_dict = checkpoint['model_cfg']

    if 'final_dim' in cfg_dict and cfg_dict['final_dim'] == 768:
        cfg_dict['final_dim'] = 1024

    cfg = Wav2Vec2Config(
        extractor_mode=cfg_dict.get('extractor_mode', 'default'),
        encoder_layers=cfg_dict.get('encoder_layers', 12),
        encoder_embed_dim=cfg_dict.get('encoder_embed_dim', 768),
        encoder_ffn_embed_dim=cfg_dict.get('encoder_ffn_embed_dim', 3072),
        encoder_attention_heads=cfg_dict.get('encoder_attention_heads', 12),
        activation_fn=cfg_dict.get('activation_fn', 'gelu'),
        dropout=cfg_dict.get('dropout', 0.1),
        attention_dropout=cfg_dict.get('attention_dropout', 0.1),
        activation_dropout=cfg_dict.get('activation_dropout', 0.1),
        final_dim=cfg_dict.get('final_dim', 1024),
        layer_norm_first=cfg_dict.get('layer_norm_first', False),
        conv_feature_layers=cfg_dict.get('conv_feature_layers', '[(512,10,5)]'),
        conv_pos=cfg_dict.get('conv_pos', 128),
        conv_pos_groups=cfg_dict.get('conv_pos_groups', 16),
        pos_conv_depth=cfg_dict.get('pos_conv_depth', 1),
        num_negatives=cfg_dict.get('num_negatives', 100),
        required_seq_len_multiple=cfg_dict.get('required_seq_len_multiple', 1)
    )

    model = Wav2Vec2Model.build_model(cfg, task=None)

    state_dict = checkpoint['model_weight']
    for key in list(state_dict.keys()):
        if 'final_proj' in key:
            del state_dict[key]

    # Load state dict in a memory-efficient way
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    return model.to(device)

# --------------------------
# 5. LoRA Implementation (Optimized)
# --------------------------
class LoRALayer(nn.Module):
    """Memory-efficient LoRA implementation"""
    def __init__(self, original_attention, rank=4):
        super().__init__()
        self.original_attention = original_attention
        self.rank = rank

        # Freeze original parameters
        for param in original_attention.parameters():
            param.requires_grad = False

        # Initialize LoRA parameters more efficiently
        # embed_dim = original_attention.embed_dim
        # self.lora_A = nn.ParameterDict({
        #     'q': nn.Parameter(torch.randn(rank, embed_dim) * 0.02,
        #     'k': nn.Parameter(torch.randn(rank, embed_dim) * 0.02,
        #     'v': nn.Parameter(torch.randn(rank, embed_dim) * 0.02
        # })

        # self.lora_B = nn.ParameterDict({
        #     'q': nn.Parameter(torch.zeros(embed_dim, rank)),
        #     'k': nn.Parameter(torch.zeros(embed_dim, rank)),
        #     'v': nn.Parameter(torch.zeros(embed_dim, rank))
        # })
        # Initialize LoRA parameters more efficiently
        embed_dim = original_attention.embed_dim
        self.lora_A = nn.ParameterDict({
            'q': nn.Parameter(torch.randn(rank, embed_dim) * 0.02),
            'k': nn.Parameter(torch.randn(rank, embed_dim) * 0.02),
            'v': nn.Parameter(torch.randn(rank, embed_dim) * 0.02)
        })

        self.lora_B = nn.ParameterDict({
            'q': nn.Parameter(torch.zeros(embed_dim, rank)),
            'k': nn.Parameter(torch.zeros(embed_dim, rank)),
            'v': nn.Parameter(torch.zeros(embed_dim, rank))
        })

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True):
        # Original attention
        attn_output, attn_weights = self.original_attention(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights
        )

        # More efficient LoRA computation
        q_lora = query @ (self.lora_A['q'].T @ self.lora_B['q'].T)
        k_lora = key @ (self.lora_A['k'].T @ self.lora_B['k'].T)
        v_lora = value @ (self.lora_A['v'].T @ self.lora_B['v'].T)

        return attn_output + q_lora + k_lora + v_lora, attn_weights

def safe_apply_lora(model, rank=4):
    """Apply LoRA more efficiently with memory management"""
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            # Replace with optimized LoRALayer
            new_layer = LoRALayer(module, rank).to(device)
            setattr(model, name, new_layer)
            # Clean up
            del module
            gc.collect()
        else:
            safe_apply_lora(module, rank)
    return model

# --------------------------
# 6. Model Components (Optimized)
# --------------------------
class SpeakerHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_speakers=100):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Linear(hidden_dim, num_speakers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.layers(x)
        return x

class ArcFaceLoss(nn.Module):
    def __init__(self, feat_dim=256, num_classes=100, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, W)
        theta = torch.acos(torch.clamp(cosine, -1+1e-7, 1-1e-7))

        one_hot = F.one_hot(labels, num_classes=self.weight.size(0))
        logits = torch.where(one_hot.bool(), theta + self.m, theta)
        logits = torch.cos(logits) * self.s

        return F.cross_entropy(logits, labels)

# --------------------------
# 7. Optimized Training Pipeline
# --------------------------
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for audio, labels in tqdm(train_loader, desc='Training'):
        try:
            audio = audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass with automatic mixed precision if on GPU
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                features = model(audio, features_only=True)['x']
                logits = model.speaker_head(features)
                loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item() * audio.size(0)
            total_samples += audio.size(0)

            # Manual memory management
            del audio, labels, features, logits
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("Memory error, skipping batch")
                continue
            raise e

    return total_loss / total_samples

def main():
    try:
        # Initialize with optimizations
        batch_size = 16 if torch.cuda.is_available() else 4
        num_workers = min(4, os.cpu_count())

        # Load model
        print("Loading model...")
        model = load_fairseq_model('model.pt')
        model = safe_apply_lora(model, rank=4)
        model.speaker_head = SpeakerHead().to(device)

        # Data loading with optimized settings
        print("Loading data...")
        root_dir = 'aac'
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset not found: {root_dir}")

        all_ids = sorted([d for d in os.listdir(root_dir) if d.startswith('id')])
        if len(all_ids) < 118:
            raise ValueError("Insufficient speaker IDs")

        train_ids = all_ids[:100]
        test_ids = all_ids[100:118]

        train_set = VoxCelebDataset(root_dir, train_ids)
        test_set = VoxCelebDataset(root_dir, test_ids)

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )

        # Loss and optimizer
        criterion = ArcFaceLoss(num_classes=len(train_ids)).to(device)
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if 'lora_' in n or 'speaker_head' in n]},
            {'params': criterion.parameters()}
        ], lr=1e-4, weight_decay=1e-5)

        # Training loop with optimizations
        max_epochs = 1
        for epoch in range(max_epochs):
            try:
                print(f"\nEpoch {epoch+1}/{max_epochs}")

                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

                # Validation
                model.eval()
                val_loss = 0
                val_samples = 0
                with torch.no_grad():
                    for audio, labels in test_loader:
                        audio = audio.to(device)
                        labels = labels.to(device)

                        features = model(audio, features_only=True)['x']
                        logits = model.speaker_head(features)
                        val_loss += criterion(logits, labels).item() * audio.size(0)
                        val_samples += audio.size(0)

                print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss/val_samples:.4f}')

            except Exception as e:
                print(f"Error in epoch {epoch+1}: {str(e)}")
                break

        # Save model
        torch.save({
            'model_state': model.state_dict(),
            'speaker_head_state': model.speaker_head.state_dict(),
            'arcface_state': criterion.state_dict()
        }, 'fine_tuned_model.pt')

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Training complete")

if __name__ == '__main__':
    print("Starting training with optimizations...")
    main()



"""For II Compare the performance of the pre-trained and fine-tuned model on the list of trial pairs - VoxCeleb1 (cleaned) dataset"""



import os
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from transformers import Wav2Vec2Processor
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model

# --- Settings ---
TRIAL_FILE = "VoxCeleb1.txt"
WAV_BASE_PATH = r"vox1/vox1_test_wav/wav"
PRETRAINED_MODEL_PATH = "model.pt"
FINETUNED_MODEL_PATH = "fine_tuned_model.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CUDA config
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.max_split_size_mb = 128
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --- Set torchaudio backend ---
try:
    torchaudio.set_audio_backend("soundfile")
except:
    print("Warning: torchaudio backend could not be set to 'soundfile'. Try: pip install soundfile")

# --- Processor (not used in Fairseq flow, but retained for potential future use) ---
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

# --- Load full fairseq model ---
def load_fairseq_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg_dict = checkpoint['model_cfg']

    if 'final_dim' in cfg_dict and cfg_dict['final_dim'] == 768:
        cfg_dict['final_dim'] = 1024

    cfg = Wav2Vec2Config(
        extractor_mode=cfg_dict.get('extractor_mode', 'default'),
        encoder_layers=cfg_dict.get('encoder_layers', 12),
        encoder_embed_dim=cfg_dict.get('encoder_embed_dim', 768),
        encoder_ffn_embed_dim=cfg_dict.get('encoder_ffn_embed_dim', 3072),
        encoder_attention_heads=cfg_dict.get('encoder_attention_heads', 12),
        activation_fn=cfg_dict.get('activation_fn', 'gelu'),
        dropout=cfg_dict.get('dropout', 0.1),
        attention_dropout=cfg_dict.get('attention_dropout', 0.1),
        activation_dropout=cfg_dict.get('activation_dropout', 0.1),
        final_dim=cfg_dict.get('final_dim', 1024),
        layer_norm_first=cfg_dict.get('layer_norm_first', False),
        conv_feature_layers=cfg_dict.get('conv_feature_layers', '[(512,10,5)]'),
        conv_pos=cfg_dict.get('conv_pos', 128),
        conv_pos_groups=cfg_dict.get('conv_pos_groups', 16),
        pos_conv_depth=cfg_dict.get('pos_conv_depth', 1),
        num_negatives=cfg_dict.get('num_negatives', 100),
        required_seq_len_multiple=cfg_dict.get('required_seq_len_multiple', 1)
    )

    model = Wav2Vec2Model.build_model(cfg, task=None)
    state_dict = checkpoint['model_weight']

    # Remove final projection if exists
    for key in list(state_dict.keys()):
        if 'final_proj' in key:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    return model

# --- Load fine-tuned model with only state_dict ---
def load_fine_tuned_model(fine_tuned_path, pretrained_path):
    base_model = load_fairseq_model(pretrained_path).to(device)
    fine_tuned_weights = torch.load(fine_tuned_path, map_location='cpu')
    base_model.load_state_dict(fine_tuned_weights, strict=False)
    return base_model

# --- Get embedding from audio ---
def get_embedding(wav_path, model):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    try:
        waveform, sr = torchaudio.load(wav_path)
    except Exception as e:
        raise RuntimeError(f"Could not load audio file: {wav_path} | Error: {e}")

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    waveform = waveform.to(device)

    with torch.no_grad():
        output = model.extract_features(waveform, padding_mask=None)
        features = output['x'] if isinstance(output, dict) else output

    return features.mean(dim=1).squeeze().cpu()

# --- Cosine similarity ---
def cosine_sim(e1, e2):
    return F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()

# --- Compute EER ---
def compute_eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100

# --- Compute TAR@1% FAR ---
def compute_tar_far(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    try:
        idx = next(i for i, val in enumerate(fpr) if val > 0.01)
        tar = tpr[idx - 1]
    except:
        tar = 0.0
    return tar

# --- Evaluate model on trial pairs ---
def evaluate_model_on_trials(model_path, is_finetuned=False):
    print(f"\nEvaluating: {model_path}")

    if is_finetuned:
        model = load_fine_tuned_model(model_path, PRETRAINED_MODEL_PATH)
    else:
        model = load_fairseq_model(model_path)
    model = model.to(device)
    model.eval()

    embeddings = {}
    y_true = []
    y_score = []

    with open(TRIAL_FILE, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        try:
            label, path1, path2 = line.strip().split()
            label = int(label)

            full_path1 = os.path.join(WAV_BASE_PATH, path1.replace('/', os.sep))
            full_path2 = os.path.join(WAV_BASE_PATH, path2.replace('/', os.sep))

            if full_path1 not in embeddings:
                embeddings[full_path1] = get_embedding(full_path1, model)
            if full_path2 not in embeddings:
                embeddings[full_path2] = get_embedding(full_path2, model)

            emb1 = embeddings[full_path1]
            emb2 = embeddings[full_path2]

            score = cosine_sim(emb1, emb2)
            y_true.append(label)
            y_score.append(score)

        except Exception as e:
            print(f"Error processing pair: {line.strip()} | {e}")

    eer = compute_eer(y_true, y_score)
    tar = compute_tar_far(y_true, y_score)

    print(f"EER: {eer:.2f}%")
    print(f"TAR@1%FAR: {tar:.3f}")

evaluate_model_on_trials(PRETRAINED_MODEL_PATH, is_finetuned=False)
evaluate_model_on_trials(FINETUNED_MODEL_PATH, is_finetuned=True)





######################### For Part III #########################################


# Cell 1: Imports
import os
import glob
import subprocess
import csv
from tqdm import tqdm
from pathlib import Path

# Cell 2: Metadata parsing function
def parse_metadata(txt_file):
    """Parse VoxCeleb2 metadata from .txt files"""
    metadata = {
        'id': None,
        'reference': None,
        'offset': 0.0,
        'duration': 0.0
    }

    with open(txt_file, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    for line in lines:
        if line.startswith('Identity'):
            metadata['id'] = line.split(':')[-1].strip()
        elif line.startswith('Reference'):
            metadata['reference'] = line.split(':')[-1].strip()
        elif line.startswith('Offset'):
            metadata['offset'] = float(line.split(':')[-1].strip())
        elif line.startswith('FRAME'):
            frame_count = len(lines) - lines.index(line) - 1
            metadata['duration'] = frame_count * 0.01  # 10ms per frame

    return metadata

# Cell 3: Configuration (modify these paths as needed)
config = {
    'aac_root': 'vox2/vox2_test_aac/aac',
    'txt_root': 'vox2/vox2_test_txt/txt',
    'output_wav': 'vox2/wav',
    'output_meta_train': 'vox2/metadata/vox_metadata_train.csv',
    'output_meta_test': 'vox2/metadata/vox_metadata_test.csv'
}

# Cell 4: Main processing function
def prepare_voxdata(config):
    # Create directories
    os.makedirs(config['output_wav'], exist_ok=True)
    os.makedirs(os.path.dirname(config['output_meta_train']), exist_ok=True)
    os.makedirs(os.path.dirname(config['output_meta_test']), exist_ok=True)

    # Get sorted speaker IDs
    speaker_ids = sorted([d for d in os.listdir(config['aac_root'])
                      if os.path.isdir(os.path.join(config['aac_root'], d))])
    train_ids = speaker_ids[:50]  # First 50 IDs for train
    test_ids = speaker_ids[-50:]  # Last 50 IDs for test

    # Process train and test data
    for mode, ids in [('train', train_ids), ('test', test_ids)]:
        output_meta = config['output_meta_train'] if mode == 'train' else config['output_meta_test']

        with open(output_meta, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['speaker_id', 'filepath', 'offset', 'duration'])

            for speaker in tqdm(ids, desc=f"Processing {mode} speakers"):
                # Convert M4A to WAV
                m4a_files = glob.glob(os.path.join(config['aac_root'], speaker, '**', '*.m4a'),
                                    recursive=True)
                for m4a_path in m4a_files:
                    wav_path = m4a_path.replace(config['aac_root'], config['output_wav']).replace('.m4a', '.wav')
                    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                    subprocess.run([
                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                        '-i', m4a_path, '-ar', '16000', '-ac', '1', wav_path
                    ])

                # Process metadata
                txt_files = glob.glob(os.path.join(config['txt_root'], speaker, '**', '*.txt'),
                                 recursive=True)
                for txt_path in txt_files:
                    meta = parse_metadata(txt_path)
                    if None in meta.values():
                        continue

                    wav_file = txt_path.replace(config['txt_root'], config['output_wav']).replace('.txt', '.wav')
                    if os.path.exists(wav_file):
                        writer.writerow([
                            meta['id'],
                            os.path.abspath(wav_file),
                            meta['offset'],
                            meta['duration']
                        ])
    print("Data preparation complete!")

# Cell 5: Execute the processing
prepare_voxdata(config)



# Cell 1: Imports
import os
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path

# Cell 2: Configuration (modify these paths as needed)
config = {
    'librispeech_dir': 'vox2/wav',
    'metadata_dir': 'vox2/metadata',
    'librimix_outdir': 'vox_mixtures',
    'n_src': 2,
    'freqs': ['16k'],
    'modes': ['max'],
    'types': ['mix_clean']
}

# Cell 3: Main processing function
def create_mixtures(config):
    # Load metadata
    meta_train = pd.read_csv(os.path.join(config['metadata_dir'], 'vox_metadata_train.csv'))
    meta_test = pd.read_csv(os.path.join(config['metadata_dir'], 'vox_metadata_test.csv'))

    # Create output structure
    (Path(config['librimix_outdir'])/'train').mkdir(parents=True, exist_ok=True)
    (Path(config['librimix_outdir'])/'test').mkdir(parents=True, exist_ok=True)

    # Process both train and test data
    for mode, meta in [('train', meta_train), ('test', meta_test)]:
        print(f"Processing {mode} data...")
        mixtures = []
        for idx in range(len(meta)//2):
            spk1, spk2 = meta.iloc[2*idx], meta.iloc[2*idx+1]

            # Load and mix audio
            sig1, sr = sf.read(spk1['filepath'])
            sig2, _ = sf.read(spk2['filepath'])

            # Align lengths
            max_len = max(len(sig1), len(sig2))
            sig1 = np.pad(sig1, (0, max_len - len(sig1)))
            sig2 = np.pad(sig2, (0, max_len - len(sig2)))

            # Apply SNR (0dB)
            mixed = sig1 + sig2

            # Save files
            mix_id = f"mix_{idx:04d}"
            out_dir = Path(config['librimix_outdir'])/mode/mix_id
            out_dir.mkdir(exist_ok=True)

            sf.write(out_dir/'mixture.wav', mixed, sr)
            sf.write(out_dir/'s1.wav', sig1, sr)
            sf.write(out_dir/'s2.wav', sig2, sr)

            if idx % 100 == 0:
                print(f"Processed {idx} {mode} mixtures")

        print(f"Completed processing {len(meta)//2} {mode} mixtures")

# Cell 4: Execute the processing
create_mixtures(config)
print("Mixture creation complete!")



#############III- A For Separation of Speakers using Sepformer###################



import os
from pathlib import Path
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator
from tqdm import tqdm

def resample_to_8k(input_path, output_path):
    """Convert audio file to 8kHz using torchaudio"""
    waveform, sample_rate = torchaudio.load(input_path)
    if sample_rate != 8000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=8000
        )
        waveform = resampler(waveform)
    torchaudio.save(output_path, waveform, 8000)

def separate_mixtures(input_root, output_root):
    # Initialize SepFormer model
    model = separator.from_hparams(
        source="speechbrain/sepformer-whamr",
        savedir='pretrained_models/sepformer-whamr'
    )

    # Create output directory
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # Find all mixture.wav files
    mixture_files = list(Path(input_root).glob('**/mixture.wav'))

    for mix_path in tqdm(mixture_files, desc="Processing mixtures"):
        # Create corresponding output directory
        rel_path = mix_path.relative_to(input_root).parent
        output_dir = Path(output_root) / rel_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Paths for 8kHz files
        mix_8k_path = output_dir / "mixture.wav"
        sep1_8k_path = output_dir / "sep_s1.wav"
        sep2_8k_path = output_dir / "sep_s2.wav"

        # 1. Convert mixture to 8kHz and save
        resample_to_8k(mix_path, mix_8k_path)

        # 2. Separate at 8kHz
        est_sources = model.separate_file(path=str(mix_8k_path))

        # 3. Save separated files at 8kHz
        torchaudio.save(
            str(sep1_8k_path),
            est_sources[:, :, 0].detach().cpu(),
            8000
        )
        torchaudio.save(
            str(sep2_8k_path),
            est_sources[:, :, 1].detach().cpu(),
            8000
        )

if __name__ == "__main__":
    # Configure paths
    input_root = "vox_mixtures/test"  # Input directory with original mixtures
    output_root = "vox_separated_8k"  # Output directory for 8kHz files

    # Run separation
    separate_mixtures(input_root, output_root)



##################III- A For Enhancement of Speakers using Sepformer#####################



import os
from pathlib import Path
import torch
import torchaudio
import numpy as np
from speechbrain.inference.separation import SepformerSeparation as separator
from pesq import pesq
from mir_eval.separation import bss_eval_sources
from tqdm import tqdm
import soundfile as sf
import csv

def validate_audio(waveform):
    """Audio validation with phase preservation"""
    if torch.max(torch.abs(waveform)) < 0.01:
        raise ValueError("Silent audio detected")
    if len(torch.unique(waveform)) < 100:
        raise ValueError("Audio contains artifacts")
    return waveform / (torch.max(torch.abs(waveform)) + 1e-7)

def correct_whamr_phase(signal):
    """Correct WHAMR's automatic phase inversion"""
    return -signal  # WHAMR outputs are phase-inverted

def calculate_metrics(original, enhanced, sr=8000):
    """Revised metric calculation for meaningful SIR values"""
    metrics = {'sir': 0.0, 'sar': 0.0, 'sdr': 0.0, 'pesq': 0.0}

    try:
        # Convert and match lengths
        orig = original.numpy().squeeze().astype(np.float64)
        enh = enhanced.numpy().squeeze().astype(np.float64)
        min_len = min(len(orig), len(enh))
        orig = orig[:min_len]
        enh = enh[:min_len]

        # Phase correction
        corr = np.corrcoef(orig, enh)[0,1]
        enh = -enh if corr < -0.8 else enh  # Only flip if strongly anti-correlated

        # Normalization with dither
        orig = orig / (np.max(np.abs(orig)) + 1e-7) + np.random.normal(0, 1e-10, min_len)
        enh = enh / (np.max(np.abs(enh)) + 1e-7) + np.random.normal(0, 1e-10, min_len)

        # NEW SIR CALCULATION METHOD
        # 1. Calculate residual (what was removed by enhancement)
        residual = orig - enh

        # 2. Calculate power ratios
        signal_power = np.mean(enh**2)
        interference_power = np.mean(residual**2)

        # 3. Compute SIR directly in dB
        with np.errstate(divide='ignore'):
            sir_db = 10 * np.log10(signal_power / (interference_power + 1e-10))

        # Traditional BSS metrics for others
        sdr, _, sar, _ = bss_eval_sources(orig[np.newaxis,:], enh[np.newaxis,:])

        metrics['sir'] = float(np.clip(sir_db, -5, 25))  # Wider realistic range
        metrics['sar'] = float(np.clip(sar[0], 0, 25))
        metrics['sdr'] = float(np.clip(sdr[0], -5, 25))

        # PESQ calculation
        if min_len >= 2400:
            metrics['pesq'] = float(np.clip(pesq(sr, orig, enh, 'nb'), 1.0, 4.5))

    except Exception as e:
        print(f"Metric calculation warning: {str(e)}")

    return metrics

def enhance_audio(model, input_path, output_dir):
    """Enhancement with WHAMR phase handling"""
    try:
        # Create output directory
        output_dir.mkdir(exist_ok=True)

        # Load and validate
        waveform, sr = torchaudio.load(input_path)
        if sr != 8000:
            raise ValueError(f"Expected 8kHz audio, got {sr}Hz")

        waveform = validate_audio(waveform)

        # Save original (with phase correction)
        spk = input_path.stem.split('_')[-1]
        orig_path = output_dir / f"orig_{spk}.wav"
        corrected_orig = correct_whamr_phase(waveform.numpy())
        sf.write(orig_path, corrected_orig.squeeze(), 8000, subtype='PCM_16')

        # Enhance
        enhanced = model.separate_file(path=str(input_path))
        enhanced = enhanced[:,:,0].detach().cpu()
        enhanced = validate_audio(enhanced)

        # Save enhanced
        enh_path = output_dir / f"enhanced_{spk}.wav"
        sf.write(enh_path, enhanced.squeeze().numpy(), 8000, subtype='PCM_16')

        return enhanced

    except Exception as e:
        print(f"Processing failed for {input_path}: {str(e)}")
        return None

def process_directory(input_root, output_root):
    """Main processing pipeline"""
    try:
        model = separator.from_hparams(
            source="speechbrain/sepformer-wham-enhancement",
            savedir='pretrained_models/sepformer-wham-enhancement'
        )
    except Exception as e:
        print(f"Model initialization failed: {str(e)}")
        return

    # Setup output
    Path(output_root).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_root) / "enhancement_metrics.csv"

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['mix_id', 'speaker', 'sir_db', 'sar_db', 'sdr_db', 'pesq'])

    # Process all speaker files
    speaker_files = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.startswith('sep_') and file.endswith('.wav'):
                speaker_files.append(Path(root) / file)

    for input_path in tqdm(speaker_files, desc="Processing"):
        mix_id = input_path.parent.name
        spk = input_path.stem.split('_')[-1]

        enhanced_dir = Path(output_root) / mix_id
        enhanced = enhance_audio(model, input_path, enhanced_dir)

        if enhanced is not None:
            try:
                original, _ = torchaudio.load(input_path)
                metrics = calculate_metrics(original, enhanced)

                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        mix_id,
                        spk,
                        f"{metrics['sir']:.2f}",
                        f"{metrics['sar']:.2f}",
                        f"{metrics['sdr']:.2f}",
                        f"{metrics['pesq']:.2f}"
                    ])

            except Exception as e:
                print(f"Metrics failed for {input_path}: {str(e)}")

if __name__ == "__main__":
    input_base = "vox_separated_8k"  # Contains sep_s1.wav, sep_s2.wav
    output_base = "vox_enhanced_8k"

    process_directory(input_base, output_base)
    print(f"Processing complete. Results saved to {output_base}")



#####################III- B####################################



import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import soundfile as sf
import io
import csv
import torchaudio
from sklearn.metrics.pairwise import cosine_similarity
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# --------------------------
# 1. Configuration for Optimization
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(os.cpu_count()) if device.type == 'cpu' else None
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

# --------------------------
# 2. Memory-Efficient Audio Loading
# --------------------------
def load_audio(path, sample_rate=16000):
    """Optimized audio loading that minimizes disk usage"""
    try:
        # First try direct loading for supported formats
        if path.lower().endswith(('.wav', '.flac')):
            audio, _ = sf.read(path, dtype='float32', always_2d=False)
        else:
            # Use in-memory conversion for unsupported formats
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-ac', '1', '-ar', str(sample_rate),
                '-f', 'wav', '-'
            ]
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL
            )
            stdout, _ = process.communicate()

            # Read directly from memory
            audio, _ = sf.read(io.BytesIO(stdout), dtype='float32')

        if len(audio) == 0 or np.max(np.abs(audio)) < 0.001:
            return np.zeros(sample_rate * 3, dtype=np.float32)

        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return np.zeros(sample_rate * 3, dtype=np.float32)

# --------------------------
# 3. Speaker Identification Model
# --------------------------
class HuBERTSpeakerIdentifier:
    def __init__(self, model_path, fine_tuned=False):
        self.device = device
        self.fine_tuned = fine_tuned

        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")

        # Load model architecture
        self.model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
        # Enable output of hidden states
        self.model.config.output_hidden_states = True

        # Load custom weights with security check
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            except:
                state_dict = torch.load(model_path, map_location='cpu')

            model_state_dict = state_dict.get('model', state_dict)
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            self.model.load_state_dict(model_state_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_embedding(self, audio_path):
        try:
            # Load and validate audio
            waveform, sr = torchaudio.load(audio_path)
            if waveform.nelement() == 0:
                raise ValueError("Empty audio file")

            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            if waveform.shape[1] < 16000:
                raise ValueError("Audio too short")

            # Extract features and move them to device
            inputs = self.feature_extractor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if torch.isnan(inputs['input_values']).any():
                raise ValueError("NaN values in input")

            with torch.no_grad():
                outputs = self.model(**inputs)

            if self.fine_tuned:
                embeddings = outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states
                # Ensure hidden_states is not None
                if hidden_states is None:
                    raise ValueError("Model did not return hidden states")
                selected_layers = hidden_states[6:13]
                embeddings = torch.mean(torch.stack(selected_layers), dim=0)

            if embeddings.ndim == 3:
                embeddings = embeddings.mean(dim=1)
            return embeddings.squeeze(0).cpu().numpy()

        except Exception as e:
            print(f"Skipped {audio_path}: {str(e)}")
            return None
# --------------------------
# 4. Evaluation Pipeline
# --------------------------
def collect_enhanced_audio_files(root_dir):
    """Collect all enhanced audio files and their speaker labels"""
    audio_files = []
    speaker_labels = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith('enhanced_') and file.endswith('.wav'):
                speaker = file.split('_')[-1].split('.')[0]
                audio_files.append(os.path.join(root, file))
                speaker_labels.append(speaker)

    return audio_files, speaker_labels

def evaluate_speaker_identification(audio_files, speaker_labels, identifier, model_name):
    """Enhanced evaluation with progress tracking"""
    print(f"\nExtracting embeddings with {model_name}...")
    embeddings = []
    valid_labels = []

    for audio_path, label in tqdm(zip(audio_files, speaker_labels),
                                total=len(audio_files),
                                desc="Processing files"):
        emb = identifier.extract_embedding(audio_path)
        if emb is not None:
            embeddings.append(emb)
            valid_labels.append(label)

    if not embeddings:
        print("No valid embeddings extracted")
        return 0.0

    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(valid_labels)

    print(f"Evaluating {model_name}...")
    correct = 0

    for i in tqdm(range(len(embeddings)), desc="Calculating similarities"):
        similarities = cosine_similarity([embeddings[i]], embeddings)[0]
        similarities[i] = -np.inf  # Exclude self
        predicted_index = np.argmax(similarities)
        if labels[predicted_index] == labels[i]:
            correct += 1

    accuracy = correct / len(embeddings)
    print(f"{model_name} Rank-1 Accuracy: {accuracy:.2%}")
    return accuracy

# --------------------------
# 5. Main Execution
# --------------------------
def main():
    # Path configurations
    enhanced_audio_dir = "vox_enhanced_8k"
    pretrained_model_path = "model.pt"  # Original HuBERT large
    finetuned_model_path = "fine_tuned_model.pt"  # Fine-tuned on VoxCeleb
    results_file = "speaker_identification_results.csv"

    # Collect enhanced audio files
    audio_files, speaker_labels = collect_enhanced_audio_files(enhanced_audio_dir)
    if not audio_files:
        print("No enhanced audio files found")
        return

    print(f"Found {len(audio_files)} enhanced audio files for evaluation")

    # Initialize identifiers
    print("\nLoading pre-trained model...")
    pretrained_identifier = HuBERTSpeakerIdentifier(pretrained_model_path, fine_tuned=False)

    print("\nLoading fine-tuned model...")
    finetuned_identifier = HuBERTSpeakerIdentifier(finetuned_model_path, fine_tuned=True)

    # Evaluate both models
    pretrained_acc = evaluate_speaker_identification(
        audio_files, speaker_labels, pretrained_identifier, "Pre-trained HuBERT"
    )

    finetuned_acc = evaluate_speaker_identification(
        audio_files, speaker_labels, finetuned_identifier, "Fine-tuned HuBERT"
    )

    # Save results
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Rank-1 Accuracy"])
        writer.writerow(["Pre-trained HuBERT", f"{pretrained_acc:.4f}"])
        writer.writerow(["Fine-tuned HuBERT", f"{finetuned_acc:.4f}"])

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    print("Starting speaker identification evaluation...")
    main()



###########################Part IV Pipeline Design###########################

import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from tqdm import tqdm
from pesq import pesq
from mir_eval.separation import bss_eval_sources
import csv
from speechbrain.inference.separation import SepformerSeparation as Separator
from torch.utils.data import Dataset, DataLoader
from transformers import HubertModel
import torch.nn.functional as F

# Configuration
class Config:
    train_data_dir = "vox_mixtures/train"
    test_data_dir = "vox_mixtures/test"
    batch_size = 2
    sample_rate = 8000
    num_epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model configurations
    sep_model_source = "speechbrain/sepformer-whamr"
    sep_model_savedir = "pretrained_models/sepformer-whamr"
    enh_model_source = "speechbrain/sepformer-wham-enhancement"
    enh_model_savedir = "pretrained_models/sepformer-wham-enhancement"
    hubert_pretrained = "model.pt"
    hubert_finetuned = "fine_tuned_model.pt"

# Dataset Loader
class MultiSpeakerDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for mix_dir in sorted(os.listdir(root_dir)):
            mix_path = os.path.join(root_dir, mix_dir, "mixture.wav")
            s1_path = os.path.join(root_dir, mix_dir, "s1.wav")
            s2_path = os.path.join(root_dir, mix_dir, "s2.wav")

            if all(os.path.exists(p) for p in [mix_path, s1_path, s2_path]):
                self.samples.append({
                    'mix': mix_path,
                    's1': s1_path,
                    's2': s2_path,
                    'mix_id': mix_dir
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        def load_audio(path):
            waveform, sr = torchaudio.load(path)
            waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform
            waveform = waveform.squeeze().contiguous()
            if sr != Config.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform,
                    orig_freq=sr,
                    new_freq=Config.sample_rate
                )
            return waveform

        return {
            'mix': load_audio(sample['mix']),
            's1': load_audio(sample['s1']),
            's2': load_audio(sample['s2']),
            'mix_id': sample['mix_id']
        }

def custom_collate(batch):
    max_len = max([x['mix'].shape[-1] for x in batch])

    def pad(item):
        return F.pad(item, (0, max_len - item.shape[-1]))

    return {
        'mix': torch.stack([pad(x['mix']) for x in batch]),
        's1': torch.stack([pad(x['s1']) for x in batch]),
        's2': torch.stack([pad(x['s2']) for x in batch]),
        'mix_id': [x['mix_id'] for x in batch]
    }

# Improved Metric Calculation
def calculate_metrics(clean, enhanced, sr=8000):
    metrics = {'sdr': np.nan, 'sir': np.nan, 'sar': np.nan, 'pesq': np.nan}
    clean = clean.numpy().astype(np.float64)
    enhanced = enhanced.numpy().astype(np.float64)

    if len(clean) == 0 or len(enhanced) == 0:
        return metrics

    # Phase alignment
    try:
        corr = np.correlate(clean, enhanced, mode='full')
        lag = np.argmax(corr) - (len(enhanced)-1)
        enhanced = np.roll(enhanced, lag)
    except:
        return metrics

    # Trim to same length
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    # BSS Eval
    try:
        sdr, sir, sar, _ = bss_eval_sources(
            clean[np.newaxis, :],
            enhanced[np.newaxis, :],
            compute_permutation=False
        )
        metrics['sdr'] = sdr[0]
        metrics['sir'] = sir[0]
        metrics['sar'] = sar[0]
    except Exception as e:
        pass

    # PESQ
    try:
        metrics['pesq'] = pesq(sr, clean, enhanced, 'nb')
    except:
        pass

    # Replace inf/nan
    for k in metrics:
        if np.isinf(metrics[k]) or np.isnan(metrics[k]):
            metrics[k] = np.nan

    return metrics

# Improved Speaker Identifier
class SpeakerIdentifier(nn.Module):
    def __init__(self, hubert_path, device):
        super().__init__()
        self.device = device
        self.model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")

        try:
            state_dict = torch.load(hubert_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading HuBERT weights: {str(e)}")

        self.model = self.model.to(device)
        self.model.eval()

    def get_embedding(self, audio):
        # Resample to 16kHz for HuBERT
        audio = audio.cpu()
        audio_16k = torchaudio.functional.resample(
            audio,
            orig_freq=Config.sample_rate,
            new_freq=16000
        )

        with torch.no_grad():
            inputs = {
                'input_values': audio_16k.unsqueeze(0).to(self.device),
                'attention_mask': torch.ones_like(audio_16k).unsqueeze(0).to(self.device)
            }
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def si_snr_loss(est, target, epsilon=1e-8):
    target_zero_mean = target - torch.mean(target, dim=-1, keepdim=True)
    est_zero_mean = est - torch.mean(est, dim=-1, keepdim=True)

    power_target = torch.sum(target_zero_mean ** 2, dim=-1) + epsilon
    alpha = torch.sum(est_zero_mean * target_zero_mean, dim=-1) / power_target

    target_component = alpha.unsqueeze(-1) * target_zero_mean
    noise_component = est_zero_mean - target_component

    power_target = torch.sum(target_component ** 2, dim=-1) + epsilon
    power_noise = torch.sum(noise_component ** 2, dim=-1) + epsilon

    si_snr = 10 * torch.log10(power_target / power_noise)
    return -torch.mean(si_snr)

# Enhanced Training Loop
def train_sepformer():
    config = Config()
    device = torch.device(config.device)

    class SafeDataset(MultiSpeakerDataset):
        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            max_length = 16000 * 2  # 2-second clips
            return {k: v[:max_length] if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()}

    train_loader = DataLoader(
        SafeDataset(config.train_data_dir),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=2,
        pin_memory=True
    )

    sep_model = Separator.from_hparams(
        source=config.sep_model_source,
        savedir=config.sep_model_savedir,
        run_opts={"device": device}
    ).to(device)

    class CheckpointedEncoder(nn.Module):
        def __init__(self, original_encoder):
            super().__init__()
            self.encoder = original_encoder

        def forward(self, x):
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False)

        def _forward_impl(self, x):
            x = self.encoder.conv1d(x)
            return torch.relu(x)

    sep_model.mods.encoder = CheckpointedEncoder(sep_model.mods.encoder)

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(sep_model.parameters(), lr=1e-4)

    for epoch in range(config.num_epochs):
        sep_model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            try:
                torch.cuda.empty_cache()

                mix = batch['mix'].unsqueeze(1).to(device)
                s1 = batch['s1'].unsqueeze(1).to(device)
                s2 = batch['s2'].unsqueeze(1).to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    est_sources = sep_model.separate_batch(mix)
                    loss_s1 = si_snr_loss(est_sources[:, 0].squeeze(1), s1.squeeze(1))
                    loss_s2 = si_snr_loss(est_sources[:, 1].squeeze(1), s2.squeeze(1))
                    loss = (loss_s1 + loss_s2) / 2

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(sep_model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                del est_sources, mix, s1, s2
                progress_bar.set_postfix({'Loss': loss.item()})

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print("\nSkipping batch due to OOM")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()

    torch.save(sep_model.state_dict(), "sepformer_trained.pt")

# Enhanced Evaluation Pipeline
def evaluate_pipeline():
    config = Config()
    test_dataset = MultiSpeakerDataset(config.test_data_dir)

    sep_model = Separator.from_hparams(
        source=config.sep_model_source,
        savedir=config.sep_model_savedir,
        run_opts={"device": config.device}
    )
    enh_model = Separator.from_hparams(
        source=config.enh_model_source,
        savedir=config.enh_model_savedir,
        run_opts={"device": config.device}
    )

    id_pretrained = SpeakerIdentifier(config.hubert_pretrained, config.device)
    id_finetuned = SpeakerIdentifier(config.hubert_finetuned, config.device)

    results = []

    for sample in tqdm(test_dataset, desc="Processing samples"):
        try:
            ref_s1 = sample['s1']
            ref_s2 = sample['s2']

            mix = sample['mix'].unsqueeze(0).to(config.device)
            with torch.no_grad():
                est_sources = sep_model.separate_batch(mix).permute(0, 2, 1)
                est_sources = est_sources[0].cpu()
                est_s1, est_s2 = est_sources[0], est_sources[1]

            def enhance(wav):
                # Resample to 16kHz for enhancement
                wav_16k = torchaudio.functional.resample(
                    wav,
                    Config.sample_rate,
                    16000
                )
                enhanced_16k = enh_model.separate_batch(
                    wav_16k.unsqueeze(0).to(config.device)
                )[0].squeeze().cpu()
                # Resample back to original rate
                return torchaudio.functional.resample(
                    enhanced_16k,
                    16000,
                    Config.sample_rate
                )

            enh_s1 = enhance(est_s1)
            enh_s2 = enhance(est_s2)

            # Metric calculation with channel alignment
            metrics_1 = {
                's1': calculate_metrics(ref_s1, enh_s1),
                's2': calculate_metrics(ref_s2, enh_s2)
            }
            metrics_2 = {
                's1': calculate_metrics(ref_s1, enh_s2),
                's2': calculate_metrics(ref_s2, enh_s1)
            }

            if (metrics_1['s1']['sdr'] + metrics_1['s2']['sdr']) > \
               (metrics_2['s1']['sdr'] + metrics_2['s2']['sdr']):
                metrics = metrics_1
                final_enh = {'s1': enh_s1, 's2': enh_s2}
            else:
                metrics = metrics_2
                final_enh = {'s1': enh_s2, 's2': enh_s1}

            # Speaker Identification
            def compute_similarity(enh, ref):
                enh_emb_p = id_pretrained.get_embedding(enh)
                ref_emb_p = id_pretrained.get_embedding(ref)
                enh_emb_f = id_finetuned.get_embedding(enh)
                ref_emb_f = id_finetuned.get_embedding(ref)
                return {
                    'pretrained': np.dot(enh_emb_p, ref_emb_p.T).item(),
                    'finetuned': np.dot(enh_emb_f, ref_emb_f.T).item()
                }

            s1_sim = compute_similarity(final_enh['s1'], ref_s1)
            s2_sim = compute_similarity(final_enh['s2'], ref_s2)
            cross_sim1 = compute_similarity(final_enh['s1'], ref_s2)
            cross_sim2 = compute_similarity(final_enh['s2'], ref_s1)

            accuracy = {
                'pretrained': (
                    (s1_sim['pretrained'] > cross_sim1['pretrained']) +
                    (s2_sim['pretrained'] > cross_sim2['pretrained'])
                ) / 2,
                'finetuned': (
                    (s1_sim['finetuned'] > cross_sim1['finetuned']) +
                    (s2_sim['finetuned'] > cross_sim2['finetuned'])
                ) / 2
            }

            results.append({
                'metrics': metrics,
                'accuracy': accuracy
            })

        except Exception as e:
            print(f"Skipping sample {sample['mix_id']}: {str(e)}")

    # Robust metric aggregation
    def safe_mean(values):
        clean_values = [v for v in values if not np.isnan(v)]
        return np.mean(clean_values) if clean_values else np.nan

    avg_metrics = {
        'sdr': safe_mean([(r['metrics']['s1']['sdr'] + r['metrics']['s2']['sdr'])/2 for r in results]),
        'sir': safe_mean([(r['metrics']['s1']['sir'] + r['metrics']['s2']['sir'])/2 for r in results]),
        'sar': safe_mean([(r['metrics']['s1']['sar'] + r['metrics']['s2']['sar'])/2 for r in results]),
        'pesq': safe_mean([(r['metrics']['s1']['pesq'] + r['metrics']['s2']['pesq'])/2 for r in results]),
        'rank1_pretrained': safe_mean([r['accuracy']['pretrained'] for r in results]),
        'rank1_finetuned': safe_mean([r['accuracy']['finetuned'] for r in results])
    }

    with open('results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for k, v in avg_metrics.items():
            writer.writerow([k, f"{v:.4f}"])

    print("\nFinal Evaluation Results:")
    print(f"SDR: {avg_metrics['sdr']:.2f} dB")
    print(f"SIR: {avg_metrics['sir']:.2f} dB")
    print(f"SAR: {avg_metrics['sar']:.2f} dB")
    print(f"PESQ: {avg_metrics['pesq']:.2f}")
    print(f"Rank-1 Accuracy (Pretrained): {avg_metrics['rank1_pretrained']:.2%}")
    print(f"Rank-1 Accuracy (Finetuned): {avg_metrics['rank1_finetuned']:.2%}")

if __name__ == "__main__":
    train_sepformer()
    evaluate_pipeline()

