#!/usr/bin/env python3
"""
🎯 Realistic Edo TTS - Easy Speech Generation
Generate realistic speech audio from Edo text!
"""

import sys
import os
from pathlib import Path

def main():
    """Main function for realistic Edo TTS"""
    
    print("🎯 REALISTIC EDO SPEECH GENERATOR")
    print("=" * 55)
    print("✅ Your trained Edo TTS model with Griffin-Lim vocoder!")
    print("🎵 Generate REALISTIC speech audio from Edo text")
    print("📝 Full support for Edo characters including ẹ, ọ, tone marks")
    print("")
    print("💡 Example phrases to try:")
    print("   • ovbe          (simple word)")
    print("   • ọse           (with tone mark)") 
    print("   • oberhọmwan    (longer word)")
    print("   • sikẹ odaro    (two words)")
    print("   • imina oghọghọ (with multiple tone marks)")
    print("")
    
    while True:
        print("-" * 55)
        edo_text = input("📝 Enter Edo text (or 'quit' to exit): ").strip()
        
        if edo_text.lower() in ['quit', 'exit', 'q']:
            print("🎉 Kú dẹ! (Thank you for using Edo TTS!)")
            break
        
        if not edo_text:
            print("⚠️  Please enter some text")
            continue
        
        print(f"🎯 Generating REALISTIC speech for: '{edo_text}'")
        
        try:
            # Import the realistic TTS engine
            from edo_realistic_tts import EdoTTSInferenceWithVocoder
            from pathlib import Path
            
            # Find the best model
            model_files = list(Path(".").glob("edo_tts_best_epoch_*.pth"))
            if not model_files:
                print("❌ No trained model found!")
                print("🔧 Make sure you're in the edo-tts-model directory")
                continue
            
            latest_model = sorted(model_files)[-1]
            
            # Create realistic inference engine
            print("🚀 Loading model with Griffin-Lim vocoder...")
            tts = EdoTTSInferenceWithVocoder(latest_model, "edo_model_info.json")
            
            # Generate realistic speech
            safe_filename = edo_text.replace(' ', '_').replace('/', '_')[:20]
            filename = f"my_realistic_edo_{safe_filename}.wav"
            
            print("🎵 Synthesizing speech...")
            result = tts.synthesize_realistic_speech(edo_text, filename)
            
            if result:
                print(f"\n✅ REALISTIC SPEECH GENERATED!")
                print(f"🎵 Audio file: {filename}")
                print(f"📊 Analysis: {result['analysis_path']}")
                print(f"⏱️  Duration: {result['duration']:.2f} seconds")
                print(f"🔊 Sample rate: {result['sample_rate']} Hz")
                
                # Instructions for playing
                print(f"\n🎧 PLAY YOUR EDO SPEECH:")
                print(f"   🔊 macOS: afplay {filename}")
                print(f"   🔊 Or open {filename} in any audio player")
                
                # Try to play automatically on macOS
                try:
                    import subprocess
                    print(f"\n🎵 Playing audio automatically...")
                    subprocess.run(["afplay", filename], check=True, capture_output=True)
                    print(f"✅ Audio played successfully!")
                except:
                    print(f"💡 Use: afplay {filename} to play manually")
                
            else:
                print("❌ Failed to generate realistic speech")
                
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("🔧 Make sure all dependencies are installed")
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            print("💡 Try a shorter text or check the model files")
            import traceback
            traceback.print_exc()
        
        print("")
        continue_choice = input("🔄 Generate more realistic speech? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes', '']:
            break
    
    print("")
    print("🎉 EDO REALISTIC SPEECH SESSION COMPLETE!")
    print("📁 All realistic audio files saved in this directory")
    print("🔊 Your Edo TTS now generates REAL speech-like audio!")

if __name__ == "__main__":
    # Ensure we're in the right directory
    model_dir = Path("/Users/asheronamac/Desktop/Coding Projects/App projects/edo_app_project/edo-tts-model")
    if model_dir.exists():
        os.chdir(model_dir)
        print(f"📁 Working in: {model_dir}")
    
    main()
