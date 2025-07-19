#!/usr/bin/env python3
"""
Complete Edo TTS Training Script using Coqui TTS
This script trains a Tacotron2 model on your Edo language dataset.
"""

import os
import torch
import json
from pathlib import Path

def setup_training():
    """Initialize training environment and check requirements"""
    print("🚀 Edo TTS Training Setup")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Check files
    required_files = [
        "edo_config.json",
        "metadata.csv",
        "metadata_train.csv", 
        "metadata_val.csv"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        print("Please run simple_setup.py first!")
        return False
        
    print("✅ All required files found")
    
    # Check audio files
    wav_count = len(list(Path("wavs").glob("*.wav")))
    print(f"🎵 Audio files: {wav_count}")
    
    return True

def start_training():
    """Start the TTS training process"""
    if not setup_training():
        return
    
    try:
        # Import TTS components
        from TTS.tts.configs.tacotron2_config import Tacotron2Config
        from TTS.tts.models.tacotron2 import Tacotron2
        from TTS.trainer import Trainer, TrainerArgs
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor
        
        print("📚 TTS libraries loaded successfully")
        
        # Load configuration
        config = Tacotron2Config()
        with open("edo_config.json", 'r') as f:
            config_dict = json.load(f)
        
        # Update config with our settings
        config.update(config_dict)
        config.run_name = "edo_tts_v1"
        config.output_path = "outputs/"
        
        # Adjust for dataset size
        config.lr = 0.0005  # Lower learning rate for small dataset
        config.batch_size = 8  # Smaller batch size
        config.eval_batch_size = 4
        config.epochs = 2000  # More epochs for small dataset
        config.save_step = 500
        config.print_step = 10
        config.plot_step = 100
        
        print(f"⚙️  Training Configuration:")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Learning rate: {config.lr}")
        print(f"   Epochs: {config.epochs}")
        print(f"   Output: {config.output_path}")
        
        # Setup audio processor
        ap = AudioProcessor.init_from_config(config)
        
        # Load datasets
        train_samples, eval_samples = load_tts_samples(
            config.datasets, eval_split=True
        )
        
        print(f"📊 Dataset loaded:")
        print(f"   Training samples: {len(train_samples)}")
        print(f"   Validation samples: {len(eval_samples)}")
        
        # Initialize tokenizer
        tokenizer, config = TTSTokenizer.init_from_config(config)
        
        # Initialize model
        model = Tacotron2(config, ap, tokenizer)
        
        print(f"🧠 Model initialized:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training arguments
        trainer_args = TrainerArgs()
        
        # Initialize trainer
        trainer = Trainer(
            trainer_args,
            config,
            output_path=config.output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples
        )
        
        print("🏃‍♂️ Starting training...")
        print("📈 Monitor progress:")
        print(f"   TensorBoard: tensorboard --logdir={config.output_path}")
        print(f"   Logs: {config.output_path}/train.log")
        
        # Start training
        trainer.fit()
        
    except ImportError as e:
        print(f"❌ TTS import error: {e}")
        print("Please install: pip install coqui-tts")
        
        # Alternative: Use command line
        print("🔄 Alternative: Using command line training")
        cmd = f"""
        tts-server --list_models
        echo "Training with command line..."
        python -m TTS.bin.train_tts --config_path edo_config.json --coqpit_overrides batch_size=8 lr=0.0005
        """
        
        print("Run this command:")
        print(cmd)
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("Check logs and configuration")

if __name__ == "__main__":
    start_training()
