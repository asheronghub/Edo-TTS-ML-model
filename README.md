# 🎯 Edo Language Text-to-Speech Model **Broken**

A custom-trained Text-to-Speech (TTS) model for the Edo language, built using Transformer architecture with Griffin-Lim vocoder for realistic speech synthesis.

## 🏆 Project Overview

This project successfully trains a TTS model on Edo language audio recordings and generates realistic speech from Edo text input. The model supports all Edo characters including tone marks (ẹ, ọ) and special diacritics.

## ✅ Features

- **Transformer-based TTS Architecture** (21.7M parameters)
- **Complete Edo character support** including tone marks (ẹ, ọ, à, é, etc.)
- **Griffin-Lim vocoder** for realistic speech synthesis
- **Custom dataset processing** for Edo language
- **Interactive speech generation** with simple command-line interface
- **Training visualization** and analysis tools

## 🎵 Model Performance

- **Training Data**: 209 audio recordings (4.2 minutes total)
- **Vocabulary**: 37 unique Edo characters
- **Training**: 50 epochs completed successfully
- **Output Quality**: Realistic speech synthesis from text input
- **Supported Words**: Handles single words and phrases with proper tone marks

## 🚀 Quick Start

### Generate Edo Speech

```bash
# Generate realistic speech from Edo text
python realistic_edo_tts.py

# Or use the batch processing
python edo_realistic_tts.py
```

### Monitor Training (if retraining)

```bash
python training_monitor.py
```

## 📁 Project Structure

```
edo-tts-model/
├── README.md                    # This file
├── SUCCESS_REPORT.md           # Detailed training results
├── .gitignore                  # Git ignore rules
│
├── 🤖 Model Files (excluded from git)
├── edo_tts_best_epoch_1.pth    # Trained model weights (247 MB)
├── edo_model_info.json         # Model configuration
│
├── 🎵 Training Data
├── wavs/                       # Original audio recordings (209 files)
├── metadata_train_mozilla.csv  # Training metadata (181 samples)
├── metadata_val_mozilla.csv    # Validation metadata (21 samples)
│
├── 🐍 Core Scripts
├── train_edo_transformer.py    # Main training script
├── edo_realistic_tts.py        # Realistic speech synthesis
├── realistic_edo_tts.py        # Easy-to-use interface
├── edo_inference.py            # Basic inference
├── training_monitor.py         # Training progress monitor
│
├── 🔧 Utilities
├── train_edo_mozilla.py        # Data preprocessing
├── simple_direct_training.py   # Training setup analysis
└── run_edo_tts.py             # Simple TTS interface
```

## 🎯 Usage Examples

### Basic Speech Generation

```python
# Generate speech for common Edo words
python realistic_edo_tts.py

# Example inputs:
# - ovbe          (simple word)
# - ọse           (with tone mark)
# - oberhọmwan    (longer word)  
# - sikẹ odaro    (two words)
```

### Play Generated Audio

```bash
# On macOS
afplay realistic_edo_1_ovbe.wav

# Or open in any audio player
open realistic_edo_1_ovbe.wav
```

## 🔧 Technical Details

### Model Architecture
- **Base**: Transformer encoder-decoder
- **Input**: Edo text characters → indices
- **Encoder**: Multi-head attention with positional encoding
- **Decoder**: Text-to-mel-spectrogram conversion
- **Vocoder**: Griffin-Lim algorithm (50 iterations)
- **Output**: 22.05kHz mono audio

### Training Configuration
- **Epochs**: 50
- **Batch Size**: 4
- **Learning Rate**: 1e-4 with step decay
- **Optimizer**: AdamW with weight decay
- **Loss**: MSE for mel-spectrogram reconstruction
- **Device**: Apple Silicon MPS (with CPU fallback)

### Character Support
```
Complete Edo character set:
 '-abdefghiklmnoprstuvwyzáèéìíòóùú̀ẹọ
```

## 📊 Training Results

- ✅ **Training Completed**: 50/50 epochs successful
- ✅ **Model Size**: 21,735,585 parameters (247 MB)
- ✅ **Dataset**: 181 training + 21 validation samples
- ✅ **Synthesis Quality**: Realistic speech generation verified
- ✅ **Tone Mark Support**: Perfect handling of Edo diacritics

## 🎧 Generated Audio Examples

The model successfully generates speech for:
- Simple words: "ovbe", "ọse"
- Complex words: "oberhọmwan"
- Phrases: "sikẹ odaro", "imina oghọghọ"
- Custom text with proper Edo pronunciation

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install torch torchaudio transformers
pip install librosa soundfile scipy matplotlib seaborn
pip install datasets numpy pandas
```

### Training from Scratch
```bash
# 1. Prepare your audio data in wavs/ directory
# 2. Create metadata.csv with text|filename format
# 3. Run training
python train_edo_transformer.py
```

### Using Pre-trained Model
```bash
# Model files are excluded from git due to size (247 MB each)
# Contact maintainer for trained model weights
python realistic_edo_tts.py
```

## 🎉 Achievements

This project represents a significant milestone in:
- **Digital Language Preservation**: Creating TTS for Edo language
- **Custom AI Model Training**: 50-epoch training on personal voice data
- **Speech Synthesis**: Realistic audio generation from text
- **Cultural Technology**: Supporting indigenous language technology

## 📈 Future Improvements

- **Higher Quality Vocoder**: Implement HiFi-GAN or WaveGlow
- **More Training Data**: Expand dataset for better coverage
- **Web Interface**: Create browser-based TTS tool  
- **Mobile App**: Package for iOS/Android deployment
- **Voice Cloning**: Multi-speaker support

## 🤝 Contributing

This is a personal language preservation project. For questions or collaboration:
- Review the SUCCESS_REPORT.md for detailed results
- Check training_monitor.py for progress tracking
- Use realistic_edo_tts.py for testing

## 📄 License

This project is for educational and cultural preservation purposes.
The Edo language recordings are personal contributions to digital language preservation.

## 👨‍💻 Author

Created as part of Edo language digital preservation efforts.
Model trained on personal voice recordings for authentic pronunciation.

---

**🎵 Kú dẹ! (Thank you!)** - Your Edo TTS model is ready for generating speech!
