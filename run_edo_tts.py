#!/usr/bin/env python3
"""
🎯 Simple Edo TTS - Easy Model Usage
Just run this script and enter Edo text to generate speech!
"""

import sys
import os
from pathlib import Path

def main():
    """Main function for easy TTS usage"""
    
    print("🎯 EDO TEXT-TO-SPEECH GENERATOR")
    print("=" * 50)
    print("✅ Your trained Edo TTS model is ready!")
    print("🎵 Enter Edo text and generate speech audio")
    print("📝 Supports all Edo characters including ẹ, ọ, tone marks")
    print("")
    print("💡 Examples to try:")
    print("   • ovbe")
    print("   • ọse") 
    print("   • oberhọmwan")
    print("   • sikẹ odaro")
    print("   • imina oghọghọ")
    print("")
    
    while True:
        print("-" * 50)
        edo_text = input("📝 Enter Edo text (or 'quit' to exit): ").strip()
        
        if edo_text.lower() in ['quit', 'exit', 'q']:
            print("🎉 Thank you for using Edo TTS!")
            break
        
        if not edo_text:
            print("⚠️  Please enter some text")
            continue
        
        print(f"🎯 Generating speech for: '{edo_text}'")
        
        # Set MPS fallback environment variable
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Import and run inference (delayed import to handle env var)
        try:
            from edo_inference import EdoTTSInference
            import torch
            from pathlib import Path
            
            # Find the best model
            model_files = list(Path(".").glob("edo_tts_best_epoch_*.pth"))
            if not model_files:
                print("❌ No trained model found!")
                print("🔧 Make sure you're in the edo-tts-model directory")
                continue
            
            latest_model = sorted(model_files)[-1]
            
            # Create inference engine
            device = torch.device("cpu")  # Use CPU to avoid MPS issues
            tts = EdoTTSInference(latest_model, "edo_model_info.json", device=device)
            
            # Generate speech
            filename = f"my_edo_speech_{len(edo_text)}.wav"
            result = tts.synthesize(edo_text, filename)
            
            if result:
                print(f"✅ Speech generated successfully!")
                print(f"🎵 Audio saved as: {filename}")
                print(f"📊 Mel spectrogram saved as: {filename.replace('.wav', '_mel.png')}")
                print(f"⏱️  Audio length: {len(result['audio'])/22050:.2f} seconds")
                
                # Instructions for playing
                print("")
                print("🎧 To play the audio:")
                print(f"   • Open {filename} in any audio player")
                print(f"   • Or use: afplay {filename}  (macOS)")
                
            else:
                print("❌ Failed to generate speech")
                
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("🔧 Make sure all dependencies are installed")
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            print("💡 Try a shorter text or check the model files")
        
        print("")
        continue_choice = input("🔄 Generate another phrase? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes', '']:
            break
    
    print("")
    print("🎉 EDO TTS SESSION COMPLETE!")
    print("📁 All generated audio files are saved in this directory")
    print("🔧 Model files preserved for future use")

if __name__ == "__main__":
    # Ensure we're in the right directory
    model_dir = Path("/Users/asheronamac/Desktop/Coding Projects/App projects/edo_app_project/edo-tts-model")
    if model_dir.exists():
        os.chdir(model_dir)
        print(f"📁 Working in: {model_dir}")
    
    main()
