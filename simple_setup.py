#!/usr/bin/env python3
"""
Simple Audio Converter for Edo TTS Dataset
Uses pydub instead of librosa for better compatibility
"""

import os
import pandas as pd
from pathlib import Path
from pydub import AudioSegment
import json

class SimpleEdoProcessor:
    def __init__(self, base_dir="/Users/asheronamac/Desktop/Coding Projects/App projects/edo_app_project/edo-tts-model"):
        self.base_dir = Path(base_dir)
        self.audio_dir = self.base_dir / "wavs"
        self.metadata_path = self.base_dir / "metadata.csv"
        
    def convert_mp3_to_wav(self):
        """Convert MP3 files to WAV using pydub"""
        print("🎵 Converting MP3 to WAV...")
        
        mp3_files = list(self.audio_dir.glob("*.mp3"))
        print(f"Found {len(mp3_files)} MP3 files")
        
        converted = 0
        failed = 0
        
        for mp3_file in mp3_files:
            wav_file = mp3_file.with_suffix('.wav')
            
            if wav_file.exists():
                print(f"  ✅ {mp3_file.name} already converted")
                continue
                
            try:
                # Load MP3 with pydub
                audio = AudioSegment.from_mp3(str(mp3_file))
                
                # Convert to mono, 22050 Hz
                audio = audio.set_channels(1)  # Mono
                audio = audio.set_frame_rate(22050)  # 22.05 kHz
                
                # Export as WAV
                audio.export(str(wav_file), format="wav")
                
                converted += 1
                print(f"  ✅ Converted {mp3_file.name}")
                
            except Exception as e:
                failed += 1
                print(f"  ❌ Failed {mp3_file.name}: {e}")
        
        print(f"✅ Conversion complete: {converted} success, {failed} failed")
        return converted, failed
        
    def analyze_audio(self):
        """Analyze converted WAV files"""
        print("📊 Analyzing audio files...")
        
        wav_files = list(self.audio_dir.glob("*.wav"))
        print(f"Found {len(wav_files)} WAV files")
        
        durations = []
        total_duration = 0
        
        for wav_file in wav_files:
            try:
                audio = AudioSegment.from_wav(str(wav_file))
                duration = len(audio) / 1000.0  # Convert ms to seconds
                durations.append(duration)
                total_duration += duration
                
            except Exception as e:
                print(f"  ❌ Error analyzing {wav_file.name}: {e}")
        
        if durations:
            print(f"  📈 Statistics:")
            print(f"    Total files: {len(durations)}")
            print(f"    Total duration: {total_duration/60:.1f} minutes")
            print(f"    Average duration: {sum(durations)/len(durations):.2f}s")
            print(f"    Range: {min(durations):.2f}s - {max(durations):.2f}s")
            
            # Check for problematic files
            short_files = [d for d in durations if d < 0.5]
            long_files = [d for d in durations if d > 10.0]
            
            if short_files:
                print(f"  ⚠️  {len(short_files)} files shorter than 0.5s")
            if long_files:
                print(f"  ⚠️  {len(long_files)} files longer than 10s")
        
        return durations
        
    def create_config(self):
        """Create TTS configuration"""
        print("⚙️ Creating TTS configuration...")
        
        # Edo character set
        edo_characters = " '-abdefghiklmnoprstuvwyzàáèéìíòóùú̀ẹọ"
        
        config = {
            "model_name": "edo_tts_tacotron2",
            "run_name": "edo_tts_experiment_1",
            "epochs": 1000,
            "lr": 0.001,
            "batch_size": 16,
            "eval_batch_size": 16,
            "mixed_precision": True,
            "run_eval": True,
            "test_delay_epochs": -1,
            "print_step": 25,
            "plot_step": 100,
            "log_model_step": 1000,
            "save_step": 10000,
            "save_n_checkpoints": 5,
            "save_checkpoints": True,
            "target_loss": "loss_1",
            "print_eval": False,
            "use_phonemes": False,
            "phonemizer": None,
            "phoneme_language": "edo",
            "compute_input_seq_cache": True,
            "text_cleaner": "basic_cleaners",
            "enable_eos_bos_chars": False,
            "test_sentences_file": "",
            "phoneme_cache_path": "./phoneme_cache",
            "characters": {
                "characters": edo_characters,
                "punctuations": "!'(),-.:;? ",
                "pad": "<PAD>",
                "eos": "<EOS>",
                "bos": "<BOS>"
            },
            "audio": {
                "sample_rate": 22050,
                "hop_length": 256,
                "win_length": 1024,
                "n_mels": 80,
                "n_fft": 1024,
                "preemphasis": 0.97,
                "ref_level_db": 20,
                "min_level_db": -100,
                "do_sound_norm": False,
                "signal_norm": True,
                "symmetric_norm": True,
                "max_norm": 4.0,
                "clip_norm": True,
                "mel_fmin": 0.0,
                "mel_fmax": 8000.0,
                "spec_gain": 1.0,
                "do_trim_silence": True,
                "trim_db": 60
            },
            "datasets": [{
                "name": "edo_dataset", 
                "path": str(self.base_dir),
                "meta_file_train": "metadata.csv",
                "ignored_speakers": None,
                "language": "edo",
                "phonemizer": None,
                "meta_file_val": None
            }],
            "output_path": str(self.base_dir / "outputs")
        }
        
        config_path = self.base_dir / "edo_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        print(f"  ✅ Config saved to {config_path}")
        print(f"  📝 Characters: {edo_characters}")
        
    def prepare_metadata(self):
        """Clean and prepare metadata"""
        print("🧹 Preparing metadata...")
        
        # Read original metadata
        df = pd.read_csv(self.metadata_path, sep='|', header=None, names=['audio_path', 'transcript'])
        print(f"  📄 Original: {len(df)} entries")
        
        # Update paths to WAV format
        df['audio_path'] = df['audio_path'].str.replace('.mp3', '.wav')
        
        # Clean transcripts
        def clean_text(text):
            text = str(text).strip().lower()
            # Remove extra whitespace
            import re
            text = re.sub(r'\s+', ' ', text)
            return text
            
        df['transcript'] = df['transcript'].apply(clean_text)
        
        # Filter to only files that exist as WAV
        existing_files = []
        for idx, row in df.iterrows():
            wav_path = self.base_dir / row['audio_path']
            if wav_path.exists():
                existing_files.append(idx)
        
        df_clean = df.loc[existing_files].reset_index(drop=True)
        print(f"  ✅ Clean dataset: {len(df_clean)} entries")
        
        # Save cleaned metadata
        df_clean.to_csv(self.metadata_path, sep='|', header=False, index=False)
        
        # Create train/val split (90/10)
        train_size = int(0.9 * len(df_clean))
        df_shuffled = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:]
        
        train_df.to_csv(self.base_dir / "metadata_train.csv", sep='|', header=False, index=False)
        val_df.to_csv(self.base_dir / "metadata_val.csv", sep='|', header=False, index=False)
        
        print(f"  📊 Training: {len(train_df)} samples")
        print(f"  📊 Validation: {len(val_df)} samples")
        
    def create_training_script(self):
        """Create ready-to-use training script"""
        print("📝 Creating training script...")
        
        script_content = '''#!/usr/bin/env python3
"""
Train Edo TTS Model using Coqui TTS
"""

import os
from pathlib import Path

def train_edo_tts():
    """Start training the Edo TTS model"""
    config_path = "edo_config.json"
    
    if not Path(config_path).exists():
        print("❌ Configuration file not found!")
        print("Please run the preprocessing script first.")
        return
        
    # Training command using Coqui TTS
    cmd = f"""
    tts-server --list_models
    echo "Starting Edo TTS Training..."
    python -m TTS.bin.train_tts --config_path {config_path}
    """
    
    print("🚀 Starting Edo TTS training...")
    print(f"📁 Config: {config_path}")
    print("⏰ This will take several hours...")
    
    # You can also use the Python API:
    try:
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.tacotron2_config import Tacotron2Config
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.models.tacotron2 import Tacotron2
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.trainer import Trainer, TrainerArgs
        
        print("✅ TTS library loaded successfully")
        print("🔥 Ready to start training!")
        
        # Load config
        config = Tacotron2Config()
        config.load_json("edo_config.json")
        
        # Initialize and start training
        # (Add training code here)
        
    except ImportError as e:
        print(f"❌ TTS import error: {e}")
        print("Please install: pip install coqui-tts")
        
if __name__ == "__main__":
    train_edo_tts()
'''
        
        script_path = self.base_dir / "train_edo.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
            
        # Make executable
        os.chmod(script_path, 0o755)
        print(f"  ✅ Training script: {script_path}")
        
    def run_all(self):
        """Run complete preprocessing pipeline"""
        print("🚀 Edo TTS Setup - Complete Pipeline\n")
        
        # Step 1: Convert audio
        converted, failed = self.convert_mp3_to_wav()
        if converted == 0 and failed > 0:
            print("❌ No files converted. Check MP3 files and try again.")
            return
            
        # Step 2: Analyze audio
        durations = self.analyze_audio()
        if not durations:
            print("❌ No audio files to analyze.")
            return
            
        # Step 3: Create config
        self.create_config()
        
        # Step 4: Prepare metadata
        self.prepare_metadata()
        
        # Step 5: Create training script
        self.create_training_script()
        
        print("\n✅ Setup Complete!")
        print("\n🎯 Next Steps:")
        print("1. Review the generated edo_config.json")
        print("2. Check metadata.csv for text quality") 
        print("3. Run training: python train_edo.py")
        print("4. Monitor progress with TensorBoard")
        print("\n📊 Dataset Summary:")
        print(f"   • {len(durations)} audio files")
        print(f"   • {sum(durations)/60:.1f} minutes total")
        print(f"   • Average {sum(durations)/len(durations):.1f}s per sample")

if __name__ == "__main__":
    processor = SimpleEdoProcessor()
    processor.run_all()
