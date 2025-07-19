# Edo TTS Model Training - Complete Guide

## Overview
This guide will help you train a Text-to-Speech (TTS) model on your 209 Edo language voice recordings.

## Prerequisites & Installation

### 1. Install Python Dependencies
```bash
# Install audio processing libraries
pip install librosa soundfile pandas numpy

# Install TTS framework (Coqui TTS - recommended)
pip install coqui-tts

# Alternative: Install specific version for stability
pip install coqui-tts==0.20.6

# Install additional dependencies
pip install torch torchaudio
pip install matplotlib tensorboard
```

### 2. System Dependencies (macOS)
```bash
# Install ffmpeg for audio processing
brew install ffmpeg

# Install other audio tools if needed
brew install sox
```

## Training Pipeline Steps

### Phase 1: Data Preprocessing ✅
Run the preprocessing script to prepare your data:

```bash
cd "/Users/asheronamac/Desktop/Coding Projects/App projects/edo_app_project/edo-tts-model"
python setup_training.py
```

This will:
- Convert MP3 → WAV (22050 Hz, mono)
- Analyze audio quality
- Create TTS configuration
- Clean metadata
- Split dataset (train/validation)

### Phase 2: Install TTS Framework
Choose one of these TTS frameworks:

#### Option A: Coqui TTS (Recommended)
```bash
# Install Coqui TTS
pip install coqui-tts

# Verify installation
tts --help
```

#### Option B: Espnet
```bash
pip install espnet
pip install espnet_tts
```

### Phase 3: Training Setup

#### Create Training Script
```python
# train_edo_tts.py
import os
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.trainer import Trainer, TrainerArgs

# Configuration
config = Tacotron2Config(
    batch_size=16,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path="phoneme_cache",
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    max_seq_len=1000000,
    output_path="output/",
    datasets=[{
        "name": "edo_dataset",
        "path": "./",
        "meta_file_train": "metadata_train.csv",
        "meta_file_val": "metadata_val.csv"
    }]
)

# Initialize model
model = Tacotron2(config, ap, tokenizer, speaker_manager=None)

# Training arguments
args = TrainerArgs()
trainer = Trainer(args, config, output_path="output/", 
                 model=model, train_samples=train_samples, 
                 eval_samples=eval_samples)

# Start training
trainer.fit()
```

### Phase 4: Monitor Training

#### Using TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir=output/

# Open browser to http://localhost:6006
```

#### Training Metrics to Watch:
- Loss curves (training & validation)
- Attention alignment plots
- Audio samples quality
- Training time per epoch

### Phase 5: Model Evaluation & Testing

#### Generate Test Audio
```python
# test_model.py
from TTS.api import TTS

# Load trained model
tts = TTS(model_path="output/best_model.pth", 
          config_path="output/config.json")

# Test with Edo text
test_sentences = [
    "amẹ odidọn",
    "ebaan wẹ miẹn mwẹn a",
    "mwaan ọnrẹn ne a rẹn vbene ọ tan sẹ hẹẹ"
]

for i, sentence in enumerate(test_sentences):
    tts.tts_to_file(text=sentence, 
                   file_path=f"test_output_{i}.wav")
```

## Expected Timeline & Resources

### Training Time Estimates:
- **CPU Only**: 3-7 days for 1000 epochs
- **GPU (recommended)**: 6-12 hours for 1000 epochs
- **Dataset size**: 209 samples (~3-4 hours total audio)

### Hardware Requirements:
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, GPU with 8GB VRAM
- **Storage**: 5-10GB for models and outputs

### Model Quality Expectations:
- **Epoch 100-200**: Basic intelligibility
- **Epoch 500-700**: Good quality
- **Epoch 800-1000**: Production ready

## Troubleshooting Common Issues

### 1. Audio Quality Issues
```bash
# Check audio sample rates
ffprobe -v quiet -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 wavs/*.wav | sort | uniq -c
```

### 2. Memory Issues
- Reduce batch_size from 16 to 8 or 4
- Use mixed precision training
- Process shorter audio clips

### 3. Character Set Issues
- Verify all Edo characters are in config
- Check for Unicode normalization
- Handle special characters (tone marks)

### 4. Training Convergence
- Monitor attention alignment plots
- Adjust learning rate if loss plateaus
- Increase dataset if overfitting

## Advanced Optimizations

### Multi-Speaker Support (Future)
```python
# If you get more speakers
config.use_speaker_embedding = True
config.num_speakers = 2  # or more
```

### Voice Cloning Setup
```python
# For voice cloning capabilities
config.use_speaker_embedding = True
config.use_gst = True  # Global Style Tokens
```

### Production Deployment
```python
# Optimize for inference
config.mixed_precision = True
model = model.half()  # Use FP16
```

## File Structure After Setup
```
edo-tts-model/
├── wavs/                    # Audio files (.wav)
├── metadata.csv            # Original metadata
├── metadata_train.csv      # Training split
├── metadata_val.csv        # Validation split
├── edo_tts_config.json     # TTS configuration
├── setup_training.py       # Preprocessing script
├── train_edo_tts.py        # Training script
├── test_model.py           # Testing script
├── output/                 # Training outputs
│   ├── checkpoints/
│   ├── logs/
│   └── best_model.pth
└── phoneme_cache/          # Cached phonemes
```

## Next Steps After Training

1. **Model Evaluation**: Test with diverse Edo sentences
2. **Voice Quality**: Compare with original recordings
3. **Deployment**: Create inference API or app
4. **Expansion**: Add more speakers or dialects
5. **Integration**: Use in Edo language learning apps

## Getting Help

- Check TTS documentation: https://tts.readthedocs.io/
- Edo language resources for text normalization
- Audio processing best practices
- GPU optimization guides

Ready to start? Run `python setup_training.py` first!
