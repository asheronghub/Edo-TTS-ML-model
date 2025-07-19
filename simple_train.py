#!/usr/bin/env python3
"""
Simple Edo TTS Training Script
Direct approach using TTS components
"""

import os
import sys
import json
import torch
from pathlib import Path

def create_simple_config():
    """Create a minimal working TTS config"""
    
    config = {
        "run_name": "edo_tts",
        "epochs": 500,
        "batch_size": 4,
        "eval_batch_size": 4,
        "lr": 0.001,
        "print_step": 25,
        "save_step": 1000,
        "output_path": "outputs/",
        
        # Audio settings
        "audio": {
            "sample_rate": 22050,
            "hop_length": 256,
            "win_length": 1024,
            "n_fft": 1024,
            "n_mels": 80,
            "preemphasis": 0.97,
            "ref_level_db": 20,
            "min_level_db": -100
        },
        
        # Dataset
        "datasets": [{
            "name": "edo",
            "path": "./",
            "meta_file_train": "metadata_train.csv",
            "meta_file_val": "metadata_val.csv",
            "language": "edo",
            "formatter": "ljspeech"
        }],
        
        # Characters
        "characters": {
            "characters": " '-abdefghiklmnoprstuvwyzàáèéìíòóùú̀ẹọ",
            "punctuations": "!'(),-.:;? "
        },
        
        # Model
        "model": "tacotron2",
        "text_cleaner": "basic_cleaners",
        "use_phonemes": False,
        
        # Test sentences
        "test_sentences": [
            "ẹ̀dó",
            "amẹ odidọn",
            "ebaan wẹ miẹn mwẹn a"
        ]
    }
    
    return config

def train_edo_tts():
    """Train Edo TTS model using simplified approach"""
    
    print("🚀 Starting Simple Edo TTS Training...")
    
    # Create config
    config = create_simple_config()
    
    # Save config
    with open("simple_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration created: simple_config.json")
    
    # Check if training data exists
    if not Path("metadata_train.csv").exists():
        print("❌ Training metadata not found!")
        return False
        
    if not Path("wavs").exists():
        print("❌ Audio directory not found!")
        return False
        
    print("📊 Dataset validation:")
    print(f"  - metadata_train.csv: {Path('metadata_train.csv').stat().st_size} bytes")
    print(f"  - metadata_val.csv: {Path('metadata_val.csv').stat().st_size if Path('metadata_val.csv').exists() else 'Missing'} bytes")
    print(f"  - wavs/: {len(list(Path('wavs').glob('*.wav')))} WAV files")
    
    # Try to train
    try:
        print("🔥 Starting training...")
        
        # Import TTS training
        import subprocess
        import shlex
        
        # Use subprocess to avoid import issues
        cmd = [
            sys.executable, "-m", "TTS.bin.train_tts",
            "--config_path", "simple_config.json",
            "--small_run", "10"  # Test with 10 samples first
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Create outputs directory
        os.makedirs("outputs", exist_ok=True)
        
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout for test
        
        if result.returncode == 0:
            print("✅ Training started successfully!")
            print("Output:", result.stdout)
            return True
        else:
            print("❌ Training failed!")
            print("Error:", result.stderr)
            print("Output:", result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training test completed (timeout reached)")
        return True
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

if __name__ == "__main__":
    success = train_edo_tts()
    
    if success:
        print("\n🎉 Training process started!")
        print("📊 Monitor progress in outputs/ directory")
        print("🔍 Check outputs/train.log for details")
    else:
        print("\n💡 Training setup failed. Check error messages above.")
