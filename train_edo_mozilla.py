#!/usr/bin/env python3
"""
Custom training script for Edo TTS that handles our specific data format
"""

import os
import json
import pandas as pd
from pathlib import Path

def create_edo_metadata():
    """Convert our metadata to the expected format"""
    
    print("📝 Converting metadata format...")
    
    # Read our metadata
    train_data = []
    with open("metadata_train.csv", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" in line:
                audio_path, text = line.split("|", 1)
                # Remove 'wavs/' prefix for filename-only
                filename = os.path.basename(audio_path)
                train_data.append(f"{text}|{filename}")
    
    # Write in Mozilla format (text|filename)
    with open("metadata_train_mozilla.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))
    
    # Do the same for validation
    val_data = []
    if os.path.exists("metadata_val.csv"):
        with open("metadata_val.csv", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    audio_path, text = line.split("|", 1)
                    filename = os.path.basename(audio_path)
                    val_data.append(f"{text}|{filename}")
        
        with open("metadata_val_mozilla.csv", "w", encoding="utf-8") as f:
            f.write("\n".join(val_data))
    
    print(f"✅ Created Mozilla format metadata:")
    print(f"  - Train samples: {len(train_data)}")
    print(f"  - Val samples: {len(val_data)}")
    
    return len(train_data), len(val_data)

def create_working_config(train_samples, val_samples):
    """Create a working TTS configuration"""
    
    # Reduce batch size based on dataset size
    batch_size = min(4, max(1, train_samples // 20))
    
    config = {
        "run_name": "edo_tts_mozilla",
        "epochs": 200 if train_samples < 100 else 500,
        "batch_size": batch_size,
        "eval_batch_size": max(1, batch_size // 2),
        "lr": 0.001,
        "print_step": 10,
        "save_step": 100,
        "output_path": "outputs/",
        
        # Audio processing  
        "audio": {
            "sample_rate": 22050,
            "hop_length": 256,
            "win_length": 1024,
            "n_fft": 1024,
            "n_mels": 80,
            "preemphasis": 0.97,
            "ref_level_db": 20,
            "min_level_db": -100,
            "power": 1.5,
            "griffin_lim_iters": 60,
            "mel_fmin": 0,
            "mel_fmax": 8000,
            "do_trim_silence": True,
            "trim_db": 45,
            "signal_norm": True,
            "symmetric_norm": False,
            "max_norm": 4.0,
            "clip_norm": True,
            "stats_path": None
        },
        
        # Dataset using Mozilla formatter
        "datasets": [{
            "name": "edo_mozilla",
            "path": "./",
            "meta_file_train": "metadata_train_mozilla.csv",
            "meta_file_val": "metadata_val_mozilla.csv" if val_samples > 0 else "metadata_train_mozilla.csv",
            "language": "edo",
            "formatter": "mozilla"
        }],
        
        # Character set for Edo
        "characters": {
            "characters": " '-abdefghiklmnoprstuvwyzàáèéìíòóùú̀ẹọ",
            "punctuations": "!'(),-.:;? ",
            "phonemes": "",
            "is_unique": True,
            "is_sorted": True
        },
        
        # Model configuration 
        "model": "tacotron2",
        "text_cleaner": "basic_cleaners",
        "use_phonemes": False,
        "r": 2,  # Reduction factor
        
        # Training parameters
        "optimizer": "Adam",
        "weight_decay": 0.000006,
        "grad_clip": 5.0,
        "lr_scheduler": "StepLR",
        "lr_scheduler_params": {
            "step_size": max(1000, train_samples * 2),
            "gamma": 0.5
        },
        
        # Test sentences
        "test_sentences": [
            "ẹ̀dó",
            "amẹ odidọn",  
            "ebaan wẹ miẹn mwẹn a"
        ]
    }
    
    return config

def main():
    """Main training function"""
    
    print("🚀 Starting Edo TTS Training (Mozilla Format)")
    print("=" * 50)
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Convert metadata format
    train_count, val_count = create_edo_metadata()
    
    # Create configuration
    config = create_working_config(train_count, val_count)
    
    # Save config
    with open("edo_mozilla_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created configuration: edo_mozilla_config.json")
    print(f"📊 Training setup:")
    print(f"  - Samples: {train_count} train, {val_count} val") 
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Epochs: {config['epochs']}")
    print(f"  - Learning rate: {config['lr']}")
    
    # Validate dataset
    if not Path("wavs").exists():
        print("❌ WAV files directory missing!")
        return False
        
    wav_count = len(list(Path("wavs").glob("*.wav")))
    print(f"  - WAV files: {wav_count}")
    
    if wav_count < train_count:
        print("⚠️  Warning: Fewer WAV files than metadata entries")
    
    print("\n🔥 Starting training...")
    print("📋 This may take several hours depending on your hardware")
    print("📊 Monitor progress in outputs/ directory")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*50)
        print("✅ Setup complete! Ready to train.")
        print("\n🚀 Next step: Run the actual training:")
        print("docker run --rm -v \"$(pwd):/workspace\" -w /workspace edo-tts \\")
        print("  python -m TTS.bin.train_tts --config_path edo_mozilla_config.json")
        print("\n📊 Monitor with:")
        print("  tail -f outputs/train.log")
        print("  tensorboard --logdir=outputs/")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
