#!/usr/bin/env python3
"""
Edo TTS Training with Transformer Architecture
Direct training implementation for Edo language
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import time
from datetime import datetime

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

class EdoTTSDataset(Dataset):
    """Dataset class for Edo TTS training"""
    
    def __init__(self, metadata_file, wav_dir="wavs", sample_rate=22050, n_mels=80):
        self.wav_dir = Path(wav_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Load metadata
        self.data = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    text, filename = line.strip().split('|', 1)
                    wav_path = self.wav_dir / filename
                    if wav_path.exists():
                        self.data.append({'text': text, 'wav_path': wav_path})
        
        print(f"📊 Loaded {len(self.data)} samples from {metadata_file}")
        
        # Create character vocabulary
        all_text = " ".join([item['text'] for item in self.data])
        unique_chars = sorted(set(all_text))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(unique_chars)
        
        print(f"🔤 Vocabulary size: {self.vocab_size}")
        print(f"📝 Characters: {''.join(unique_chars)}")
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels
        )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process audio
        try:
            waveform, sr = torchaudio.load(item['wav_path'])
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Generate mel spectrogram
            mel_spec = self.mel_transform(waveform)
            mel_spec = mel_spec.squeeze(0).T  # (time, n_mels)
            
            # Convert text to indices
            text_indices = [self.char_to_idx[char] for char in item['text']]
            text_tensor = torch.tensor(text_indices, dtype=torch.long)
            
            return {
                'text': text_tensor,
                'text_length': len(text_indices),
                'mel': mel_spec,
                'mel_length': mel_spec.shape[0],
                'raw_text': item['text'],
                'filename': item['wav_path'].name
            }
            
        except Exception as e:
            print(f"❌ Error loading {item['wav_path']}: {e}")
            # Return a dummy sample
            return {
                'text': torch.tensor([0], dtype=torch.long),
                'text_length': 1,
                'mel': torch.zeros(10, self.n_mels),
                'mel_length': 10,
                'raw_text': '',
                'filename': 'error.wav'
            }

def collate_batch(batch):
    """Custom collate function for DataLoader"""
    
    # Sort batch by text length (for better padding efficiency)
    batch = sorted(batch, key=lambda x: x['text_length'], reverse=True)
    
    # Separate components
    texts = [item['text'] for item in batch]
    mels = [item['mel'] for item in batch]
    text_lengths = torch.tensor([item['text_length'] for item in batch])
    mel_lengths = torch.tensor([item['mel_length'] for item in batch])
    raw_texts = [item['raw_text'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    mels_padded = pad_sequence(mels, batch_first=True, padding_value=0)
    
    return {
        'texts': texts_padded,
        'mels': mels_padded,
        'text_lengths': text_lengths,
        'mel_lengths': mel_lengths,
        'raw_texts': raw_texts,
        'filenames': filenames
    }

class EdoTTSTransformer(nn.Module):
    """Transformer-based TTS model for Edo language"""
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, n_mels=80):
        super(EdoTTSTransformer, self).__init__()
        
        self.d_model = d_model
        self.n_mels = n_mels
        
        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))  # Max length 1000
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Duration predictor (predicts how long each character should be)
        self.duration_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
            nn.Softplus()  # Ensure positive durations
        )
        
        # Mel decoder
        self.mel_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_mels)
        )
        
        # Postnet for mel refinement
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, n_mels, 5, padding=2)
        )
        
    def forward(self, texts, text_lengths):
        batch_size, seq_len = texts.shape
        
        # Text embedding with positional encoding
        embedded = self.text_embedding(texts) * (self.d_model ** 0.5)
        embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Create padding mask
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=texts.device)
        for i, length in enumerate(text_lengths):
            padding_mask[i, length:] = True
        
        # Transformer encoding
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        # Duration prediction
        durations = self.duration_predictor(encoded).squeeze(-1)  # (batch, seq_len)
        
        # Length regulation (expand encoded features based on predicted durations)
        # For training, we'll use a simplified version
        expanded_features = encoded.repeat_interleave(2, dim=1)  # Simple 2x expansion
        
        # Mel prediction
        mel_output = self.mel_decoder(expanded_features)
        
        # Postnet refinement
        mel_postnet = mel_output.transpose(1, 2)  # For conv1d (batch, n_mels, time)
        mel_refined = self.postnet(mel_postnet)
        mel_output_refined = mel_output + mel_refined.transpose(1, 2)
        
        return {
            'mel_output': mel_output,
            'mel_output_refined': mel_output_refined,
            'durations': durations,
            'encoded': encoded
        }

class EdoTTSTrainer:
    """Training class for Edo TTS"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizers and loss
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.mel_criterion = nn.MSELoss()
        self.duration_criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        print(f"\n🚂 Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move to device
                texts = batch['texts'].to(self.device)
                mels = batch['mels'].to(self.device)
                text_lengths = batch['text_lengths'].to(self.device)
                mel_lengths = batch['mel_lengths'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(texts, text_lengths)
                
                # Calculate losses
                mel_output = outputs['mel_output']
                mel_refined = outputs['mel_output_refined']
                
                # Truncate outputs to match target lengths (simplified)
                max_mel_len = min(mel_output.shape[1], mels.shape[1])
                mel_loss1 = self.mel_criterion(
                    mel_output[:, :max_mel_len, :], 
                    mels[:, :max_mel_len, :]
                )
                mel_loss2 = self.mel_criterion(
                    mel_refined[:, :max_mel_len, :], 
                    mels[:, :max_mel_len, :]
                )
                
                total_loss_batch = mel_loss1 + mel_loss2
                
                # Backward pass
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  📊 Batch {batch_idx:3d}: Loss = {total_loss_batch.item():.4f}")
                
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        
        print(f"✅ Epoch {epoch + 1} - Average Train Loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        print(f"🔍 Validating Epoch {epoch + 1}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    # Move to device  
                    texts = batch['texts'].to(self.device)
                    mels = batch['mels'].to(self.device)
                    text_lengths = batch['text_lengths'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(texts, text_lengths)
                    
                    # Calculate loss
                    mel_output = outputs['mel_output_refined']
                    max_mel_len = min(mel_output.shape[1], mels.shape[1])
                    val_loss = self.mel_criterion(
                        mel_output[:, :max_mel_len, :], 
                        mels[:, :max_mel_len, :]
                    )
                    
                    total_loss += val_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"❌ Validation error in batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.val_losses.append(avg_loss)
        
        print(f"✅ Epoch {epoch + 1} - Validation Loss: {avg_loss:.4f}")
        return avg_loss
    
    def train(self, num_epochs=100):
        """Full training loop"""
        
        print("🎯 Starting Edo TTS Training")
        print(f"🔧 Device: {self.device}")
        print(f"📊 Training batches: {len(self.train_loader)}")
        print(f"📊 Validation batches: {len(self.val_loader)}")
        print(f"🎯 Epochs: {num_epochs}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            if self.val_loader:
                val_loss = self.validate(epoch)
            else:
                val_loss = train_loss
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"edo_tts_best_epoch_{epoch+1}.pth")
                print(f"💾 Saved best model (loss: {val_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f"edo_tts_checkpoint_epoch_{epoch+1}.pth")
            
            epoch_time = time.time() - start_time
            print(f"⏱️  Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
            print("-" * 60)
            
        print("🎉 Training completed!")
        self.plot_losses()
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(checkpoint, filename)
    
    def plot_losses(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses[-50:], label='Training Loss (Last 50 epochs)')
        if self.val_losses:
            plt.plot(self.val_losses[-50:], label='Validation Loss (Last 50 epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Recent Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('edo_tts_training_curves.png', dpi=300, bbox_inches='tight')
        print("📊 Saved training curves to edo_tts_training_curves.png")

def main():
    """Main training function"""
    
    # Check for data files
    train_file = "metadata_train_mozilla.csv"
    val_file = "metadata_val_mozilla.csv" 
    
    if not Path(train_file).exists():
        print(f"❌ Training file {train_file} not found!")
        print("🔧 Run: python train_edo_mozilla.py first")
        return
    
    if not Path("wavs").exists():
        print("❌ WAV directory not found!")
        return
    
    # Create datasets
    print("📊 Creating datasets...")
    train_dataset = EdoTTSDataset(train_file)
    
    if Path(val_file).exists():
        val_dataset = EdoTTSDataset(val_file)
    else:
        # Split training data for validation
        print("⚠️  No validation file found, using 10% of training data")
        val_size = len(train_dataset) // 10
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_batch,
        num_workers=0  # Set to 0 for debugging
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collate_batch,
        num_workers=0
    ) if val_dataset else None
    
    # Create model
    vocab_size = train_dataset.dataset.vocab_size if hasattr(train_dataset, 'dataset') else train_dataset.vocab_size
    model = EdoTTSTransformer(vocab_size=vocab_size)
    
    print(f"🤖 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = EdoTTSTrainer(model, train_loader, val_loader, device)
    
    # Save model info
    model_info = {
        'vocab_size': vocab_size,
        'char_to_idx': train_dataset.dataset.char_to_idx if hasattr(train_dataset, 'dataset') else train_dataset.char_to_idx,
        'idx_to_char': train_dataset.dataset.idx_to_char if hasattr(train_dataset, 'dataset') else train_dataset.idx_to_char,
        'model_architecture': 'Transformer',
        'n_mels': 80,
        'sample_rate': 22050
    }
    
    with open('edo_model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print("💾 Saved model info to edo_model_info.json")
    
    # Start training
    trainer.train(num_epochs=50)  # Start with 50 epochs
    
    print("\n🎉 EDO TTS TRAINING COMPLETED!")
    print("📁 Files created:")
    print("  🤖 edo_tts_best_epoch_X.pth - Best model weights")
    print("  📊 edo_tts_training_curves.png - Training visualization")
    print("  📝 edo_model_info.json - Model configuration")

if __name__ == "__main__":
    main()
