#!/usr/bin/env python3
"""
Edo TTS Training Monitor
Monitor the progress of your Edo TTS training
"""

import os
import time
import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def monitor_training():
    """Monitor the training progress"""
    
    print("🔍 Edo TTS Training Monitor")
    print("=" * 40)
    
    # Check if training is running
    training_running = False
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'train_edo_transformer.py' in ' '.join(cmdline):
                    training_running = True
                    print(f"✅ Training process found (PID: {proc.info['pid']})")
                    break
            except:
                continue
    except ImportError:
        print("💡 Install psutil for process monitoring: pip install psutil")
    
    if not training_running:
        print("⚠️  Training process not detected")
    
    # Check for model files
    model_files = list(Path(".").glob("edo_tts_best_epoch_*.pth"))
    checkpoint_files = list(Path(".").glob("edo_tts_checkpoint_epoch_*.pth"))
    
    print(f"\n📊 Training Progress:")
    print(f"  🏆 Best models: {len(model_files)}")
    print(f"  💾 Checkpoints: {len(checkpoint_files)}")
    
    # Find latest model
    if model_files:
        latest_model = sorted(model_files)[-1]
        model_size = latest_model.stat().st_size / (1024*1024)  # MB
        model_time = time.ctime(latest_model.stat().st_mtime)
        
        # Extract epoch number from filename
        epoch_num = latest_model.name.split('_')[-1].split('.')[0]
        
        print(f"  📈 Latest best model: Epoch {epoch_num}")
        print(f"  📁 File size: {model_size:.1f} MB")
        print(f"  ⏰ Last updated: {model_time}")
        
        # Try to load and show training info
        try:
            checkpoint = torch.load(latest_model, map_location='cpu')
            if 'train_losses' in checkpoint:
                train_losses = checkpoint['train_losses']
                val_losses = checkpoint.get('val_losses', [])
                
                print(f"\n📊 Training Metrics:")
                print(f"  🔥 Epochs completed: {len(train_losses)}")
                if train_losses:
                    print(f"  📉 Latest train loss: {train_losses[-1]:.2f}")
                    print(f"  📈 Best train loss: {min(train_losses):.2f}")
                    
                    if len(train_losses) > 1:
                        loss_trend = train_losses[-1] - train_losses[-2]
                        trend_icon = "📉" if loss_trend < 0 else "📈"
                        print(f"  {trend_icon} Loss change: {loss_trend:+.2f}")
                
                # Plot training progress
                if len(train_losses) > 5:  # Only plot if we have some data
                    plt.figure(figsize=(12, 5))
                    
                    plt.subplot(1, 2, 1)
                    plt.plot(train_losses, 'b-', label='Training Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Edo TTS Training Progress')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(1, 2, 2)
                    if len(train_losses) > 10:
                        recent_losses = train_losses[-10:]
                        plt.plot(range(len(train_losses)-10, len(train_losses)), recent_losses, 'r-', label='Recent Loss')
                    else:
                        plt.plot(train_losses, 'r-', label='Training Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Recent Training Progress')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig('edo_training_monitor.png', dpi=300, bbox_inches='tight')
                    print(f"  📊 Progress plot saved: edo_training_monitor.png")
                    
        except Exception as e:
            print(f"⚠️  Could not load checkpoint details: {e}")
    
    # Check dataset info
    if Path("edo_model_info.json").exists():
        with open("edo_model_info.json", 'r') as f:
            model_info = json.load(f)
        
        print(f"\n📝 Model Configuration:")
        print(f"  🔤 Vocabulary size: {model_info.get('vocab_size', 'Unknown')}")
        print(f"  🎵 Sample rate: {model_info.get('sample_rate', 'Unknown')} Hz")
        print(f"  📊 Mel channels: {model_info.get('n_mels', 'Unknown')}")
        print(f"  🏗️  Architecture: {model_info.get('model_architecture', 'Unknown')}")
    
    # Dataset stats
    if Path("metadata_train_mozilla.csv").exists():
        with open("metadata_train_mozilla.csv", 'r') as f:
            train_lines = len(f.readlines())
    else:
        train_lines = "Unknown"
    
    if Path("metadata_val_mozilla.csv").exists():
        with open("metadata_val_mozilla.csv", 'r') as f:
            val_lines = len(f.readlines())
    else:
        val_lines = "Unknown"
    
    print(f"\n📊 Dataset Information:")
    print(f"  🎓 Training samples: {train_lines}")
    print(f"  ✅ Validation samples: {val_lines}")
    
    # Check WAV files
    if Path("wavs").exists():
        wav_files = list(Path("wavs").glob("*.wav"))
        total_duration = 0
        
        try:
            import librosa
            sample_files = wav_files[:5]  # Check first 5 files for speed
            for wav_file in sample_files:
                try:
                    duration = librosa.get_duration(path=str(wav_file))
                    total_duration += duration
                except:
                    continue
            
            if sample_files:
                avg_duration = total_duration / len(sample_files)
                estimated_total = avg_duration * len(wav_files)
                print(f"  🎵 Audio files: {len(wav_files)}")
                print(f"  ⏱️  Estimated total duration: {estimated_total/60:.1f} minutes")
                print(f"  📏 Average file length: {avg_duration:.2f} seconds")
        except ImportError:
            print(f"  🎵 Audio files: {len(wav_files)}")
            print("  💡 Install librosa for duration analysis")
    
    return {
        'training_running': training_running,
        'model_count': len(model_files),
        'checkpoint_count': len(checkpoint_files),
        'latest_model': model_files[-1] if model_files else None
    }

def show_recent_activity():
    """Show recent training activity"""
    
    print("\n📈 Recent Activity:")
    
    # Check recent files
    recent_files = []
    for pattern in ["edo_tts_*.pth", "*.png", "*.json"]:
        files = list(Path(".").glob(pattern))
        for file in files:
            mtime = file.stat().st_mtime
            recent_files.append((file, mtime))
    
    # Sort by modification time (newest first)
    recent_files.sort(key=lambda x: x[1], reverse=True)
    
    print("  📁 Recent files (last 10):")
    for file, mtime in recent_files[:10]:
        time_str = time.strftime("%H:%M:%S", time.localtime(mtime))
        size_mb = file.stat().st_size / (1024*1024)
        
        if size_mb > 100:
            size_str = f"{size_mb:.0f} MB"
        elif size_mb > 1:
            size_str = f"{size_mb:.1f} MB"
        else:
            size_str = f"{file.stat().st_size/1024:.1f} KB"
        
        print(f"    {time_str}: {file.name} ({size_str})")

def main():
    """Main monitoring function"""
    
    # Change to model directory if not already there
    model_dir = Path("/Users/asheronamac/Desktop/Coding Projects/App projects/edo_app_project/edo-tts-model")
    if model_dir.exists():
        os.chdir(model_dir)
    
    print("🎯 EDO TTS TRAINING MONITOR")
    print("=" * 50)
    print(f"📁 Monitoring directory: {Path.cwd()}")
    
    # Run monitoring
    status = monitor_training()
    
    # Show recent activity
    show_recent_activity()
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if status['training_running']:
        print("  ✅ Training is active - let it continue!")
        print("  📊 Check edo_training_monitor.png for progress visualization")
        print("  ⏰ Training will save checkpoints every 10 epochs")
    else:
        if status['model_count'] > 0:
            print("  🎯 Training completed or paused")
            print("  🧪 Test the model with: python edo_inference.py")
        else:
            print("  🚀 Start training with: python train_edo_transformer.py")
    
    print("  🔄 Run this monitor script again to check progress")
    
    # Inference instructions
    if status['latest_model']:
        print(f"\n🧪 Testing Instructions:")
        print(f"  📝 Latest model: {status['latest_model'].name}")
        print(f"  🎯 Test synthesis: python edo_inference.py")
        print(f"  🎵 This will generate audio from Edo text")

if __name__ == "__main__":
    main()
