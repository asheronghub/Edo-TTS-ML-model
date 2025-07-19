#!/usr/bin/env python3
"""
Quick Model Status Checker
Run this to see if your Edo TTS model is ready
"""

import os
from pathlib import Path

def check_model_status():
    """Check if the Edo TTS model has been trained and is ready"""
    
    print("🔍 Checking Edo TTS Model Status...\n")
    
    # Check for outputs directory
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("❌ No 'outputs' directory found")
        print("📋 Status: Training has not started or failed to create outputs")
        print("🚀 Next step: Run training with './train_with_docker.sh' or 'python train_edo.py'")
        return False
    
    print("✅ Found 'outputs' directory")
    
    # Check for essential model files
    essential_files = {
        "best_model.pth": "Main trained model",
        "config.json": "Model configuration", 
        "scale_stats.npy": "Audio preprocessing stats"
    }
    
    missing_files = []
    existing_files = []
    
    for filename, description in essential_files.items():
        file_path = outputs_dir / filename
        
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
            existing_files.append((filename, description, file_size))
            print(f"✅ {filename} - {description} ({file_size:.1f} MB)")
        else:
            missing_files.append((filename, description))
            print(f"❌ {filename} - {description} (missing)")
    
    # Check for checkpoints (indicates training progress)
    checkpoints_dir = outputs_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob("*.pth"))
        if checkpoint_files:
            print(f"📊 Found {len(checkpoint_files)} training checkpoints")
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            print(f"   Latest: {latest_checkpoint.name}")
    
    # Check for training logs
    log_file = outputs_dir / "train.log"
    if log_file.exists():
        log_size = log_file.stat().st_size / 1024  # Size in KB
        print(f"📜 Training log found ({log_size:.1f} KB)")
        
        # Try to read last few lines of log
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"   Last log entry: {lines[-1].strip()}")
        except Exception as e:
            print(f"   (Could not read log: {e})")
    
    print()
    
    # Determine overall status
    if not missing_files:
        print("🎉 MODEL IS READY!")
        print("📦 All essential files present")
        print("🧪 Run 'python test_model.py' to test your model")
        print("🎵 Your Edo TTS model is ready to use!")
        return True
        
    elif existing_files:
        print("⏳ TRAINING IN PROGRESS")
        print("📈 Some files exist but training may still be running")
        print("🔄 Check training progress with 'tail -f outputs/train.log'")
        return False
        
    else:
        print("❌ NO MODEL FILES FOUND")
        print("📋 Training may have failed or not started")
        print("🚀 Start training with './train_with_docker.sh'")
        return False

def show_next_steps():
    """Show what to do next based on model status"""
    
    print("\n" + "="*50)
    print("🎯 NEXT STEPS:")
    print("="*50)
    
    if not Path("outputs").exists():
        print("1. 🚀 START TRAINING:")
        print("   ./train_with_docker.sh     # Docker method (recommended)")
        print("   python train_edo.py        # Local method")
        print()
        print("2. ⏱️  ESTIMATED TIME: 6-24 hours")
        print("3. 📊 MONITOR PROGRESS:")
        print("   python check_model.py      # Check this status")
        print("   tail -f outputs/train.log  # Watch training logs")
        
    elif Path("outputs/best_model.pth").exists():
        print("1. 🧪 TEST YOUR MODEL:")
        print("   python test_model.py       # Generate test audio")
        print()
        print("2. 🎵 USE YOUR MODEL:")
        print("   python edo_tts_api.py      # Start API server")
        print()
        print("3. 📦 PACKAGE FOR SHARING:")
        print("   ./create_package.sh        # Create shareable package")
        
    else:
        print("1. ⏳ WAIT FOR TRAINING:")
        print("   Training appears to be in progress...")
        print("   Check status: python check_model.py")
        print()
        print("2. 📊 MONITOR:")
        print("   tail -f outputs/train.log  # Watch training")
        print("   tensorboard --logdir outputs/  # Visual monitoring")

if __name__ == "__main__":
    model_ready = check_model_status()
    show_next_steps()
    
    if model_ready:
        print("\n🎉 Congratulations! Your Edo TTS model is ready to use!")
    else:
        print(f"\n⏳ Model not ready yet. Run this script again to check progress.")
