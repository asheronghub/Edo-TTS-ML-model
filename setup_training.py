#!/usr/bin/env python3
"""
Complete TTS Training Setup for Edo Language Dataset
This script handles all preprocessing steps for TTS model training.
"""

import os
import subprocess
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import json
import shutil

class EdoTTSPreprocessor:
    def __init__(self, base_dir="/Users/asheronamac/Desktop/Coding Projects/App projects/edo_app_project/edo-tts-model"):
        self.base_dir = Path(base_dir)
        self.audio_dir = self.base_dir / "wavs"
        self.metadata_path = self.base_dir / "metadata.csv"
        self.config_path = self.base_dir / "edo_tts_config.json"
        
        # TTS model requirements
        self.target_sample_rate = 22050
        self.target_format = "wav"
        
    def step1_convert_audio_format(self):
        """Convert MP3 files to WAV format with proper sample rate"""
        print("🎵 Step 1: Converting MP3 to WAV format...")
        
        mp3_files = list(self.audio_dir.glob("*.mp3"))
        print(f"Found {len(mp3_files)} MP3 files to convert")
        
        converted_count = 0
        for mp3_file in mp3_files:
            wav_file = mp3_file.with_suffix('.wav')
            
            if wav_file.exists():
                print(f"  Skipping {mp3_file.name} (WAV already exists)")
                continue
                
            try:
                # Load audio with librosa and save as WAV
                audio, sr = librosa.load(str(mp3_file), sr=self.target_sample_rate, mono=True)
                sf.write(str(wav_file), audio, self.target_sample_rate)
                converted_count += 1
                print(f"  ✅ Converted {mp3_file.name} -> {wav_file.name}")
            except Exception as e:
                print(f"  ❌ Failed to convert {mp3_file.name}: {e}")
        
        print(f"✅ Converted {converted_count} files to WAV format\n")
        
    def step2_audio_analysis(self):
        """Analyze audio files for quality and duration"""
        print("📊 Step 2: Analyzing audio quality...")
        
        wav_files = list(self.audio_dir.glob("*.wav"))
        durations = []
        sample_rates = []
        problematic_files = []
        
        for wav_file in wav_files:
            try:
                audio, sr = librosa.load(str(wav_file), sr=None)
                duration = len(audio) / sr
                durations.append(duration)
                sample_rates.append(sr)
                
                # Check for issues
                if duration < 0.5:
                    problematic_files.append((wav_file.name, "Too short", duration))
                elif duration > 10.0:
                    problematic_files.append((wav_file.name, "Too long", duration))
                    
            except Exception as e:
                problematic_files.append((wav_file.name, "Load error", str(e)))
        
        # Statistics
        print(f"  📈 Audio Statistics:")
        print(f"    Total files: {len(wav_files)}")
        print(f"    Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
        print(f"    Average duration: {np.mean(durations):.2f}s")
        print(f"    Sample rates: {set(sample_rates)}")
        
        if problematic_files:
            print(f"  ⚠️  Problematic files ({len(problematic_files)}):")
            for name, issue, detail in problematic_files:
                print(f"    {name}: {issue} ({detail})")
        else:
            print("  ✅ All audio files look good!")
        
        print()
        return durations, problematic_files
        
    def step3_create_config(self):
        """Create TTS model configuration"""
        print("⚙️  Step 3: Creating TTS configuration...")
        
        # Edo character set (based on your previous analysis)
        edo_characters = " '-abdefghiklmnoprstuvwyzàáèéìíòóùú̀ẹọ"
        
        config = {
            "model_type": "tacotron2",
            "dataset": {
                "name": "edo_tts",
                "path": str(self.base_dir),
                "meta_file_train": "metadata.csv",
                "language": "edo",
                "characters": edo_characters,
                "phoneme_cache_path": str(self.base_dir / "phoneme_cache"),
                "sample_rate": self.target_sample_rate,
                "text_cleaner": "edo_cleaner"
            },
            "audio": {
                "sample_rate": self.target_sample_rate,
                "hop_length": 256,
                "win_length": 1024,
                "n_mels": 80,
                "n_fft": 1024,
                "preemphasis": 0.97,
                "ref_level_db": 20,
                "min_level_db": -100
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 0.001,
                "max_epochs": 1000,
                "save_step": 10000,
                "checkpoint_save_step": 10000,
                "eval_step": 1000,
                "print_step": 100
            },
            "model": {
                "r": 2,
                "memory_size": 5,
                "attention_type": "original",
                "windowing": False,
                "use_forward_attn": True,
                "forward_attn_mask": True,
                "transition_agent": False,
                "location_attn": True,
                "attention_norm": "sigmoid"
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        print(f"  ✅ Configuration saved to {self.config_path}")
        print(f"  📝 Character set: {edo_characters}")
        print()
        
    def step4_clean_metadata(self):
        """Clean and validate metadata.csv"""
        print("🧹 Step 4: Cleaning metadata...")
        
        # Read metadata
        df = pd.read_csv(self.metadata_path, sep='|', header=None, names=['audio_path', 'transcript'])
        print(f"  📄 Loaded {len(df)} entries from metadata.csv")
        
        # Update audio paths to use .wav extension
        df['audio_path'] = df['audio_path'].str.replace('.wav', '.wav').str.replace('.mp3', '.wav')
        
        # Clean transcripts (normalize, lowercase)
        def clean_text(text):
            import re
            text = str(text).strip().lower()
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            return text
            
        df['transcript'] = df['transcript'].apply(clean_text)
        
        # Validate that audio files exist
        missing_files = []
        for idx, row in df.iterrows():
            audio_path = self.base_dir / row['audio_path']
            if not audio_path.exists():
                missing_files.append(row['audio_path'])
        
        if missing_files:
            print(f"  ⚠️  Missing audio files ({len(missing_files)}):")
            for f in missing_files[:5]:  # Show first 5
                print(f"    {f}")
            if len(missing_files) > 5:
                print(f"    ... and {len(missing_files) - 5} more")
        
        # Save cleaned metadata
        df.to_csv(self.metadata_path, sep='|', header=False, index=False)
        print(f"  ✅ Cleaned metadata saved ({len(df)} entries)")
        print()
        
    def step5_split_dataset(self):
        """Split dataset into train/validation sets"""
        print("📊 Step 5: Splitting dataset...")
        
        df = pd.read_csv(self.metadata_path, sep='|', header=None, names=['audio_path', 'transcript'])
        
        # 90% train, 10% validation
        train_size = int(0.9 * len(df))
        
        # Shuffle and split
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:]
        
        # Save splits
        train_df.to_csv(self.base_dir / "metadata_train.csv", sep='|', header=False, index=False)
        val_df.to_csv(self.base_dir / "metadata_val.csv", sep='|', header=False, index=False)
        
        print(f"  📈 Training set: {len(train_df)} samples")
        print(f"  📊 Validation set: {len(val_df)} samples")
        print()
        
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("🚀 Starting Edo TTS Preprocessing Pipeline\n")
        
        self.step1_convert_audio_format()
        self.step2_audio_analysis()
        self.step3_create_config()
        self.step4_clean_metadata()
        self.step5_split_dataset()
        
        print("✅ Preprocessing complete! Ready for TTS training.")
        print("\nNext steps:")
        print("1. Install TTS framework (Coqui TTS recommended)")
        print("2. Run training with the generated configuration")
        print("3. Monitor training progress and adjust parameters")

if __name__ == "__main__":
    preprocessor = EdoTTSPreprocessor()
    preprocessor.run_preprocessing()
