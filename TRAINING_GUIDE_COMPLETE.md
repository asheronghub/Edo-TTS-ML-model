# Edo TTS Model Training - Complete Guide & Alternative Approaches

## Current Status: Ready for Training ✅

Your Edo dataset has been successfully preprocessed:
- **209 audio files** converted to WAV format (22050 Hz, mono)
- **6.2 minutes** total audio duration
- **Average 1.8 seconds** per sample
- Clean metadata with train/validation split (181/21 samples)
- Edo character set extracted and configured

## Option 1: Docker-based Training (Recommended)

Since Python 3.13 has compatibility issues with TTS libraries, use Docker:

### Step 1: Create Docker Setup
```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install TTS==0.22.0
RUN pip install tensorboard pandas numpy

WORKDIR /workspace
COPY . .

CMD ["python", "train_docker.py"]
EOF

# Create training script for Docker
cat > train_docker.py << 'EOF'
#!/usr/bin/env python3
import os
os.system("python -m TTS.bin.train_tts --config_path edo_config.json")
EOF
```

### Step 2: Build and Run
```bash
cd "/Users/asheronamac/Desktop/Coding Projects/App projects/edo_app_project/edo-tts-model"

# Build Docker image
docker build -t edo-tts .

# Run training (will take several hours)
docker run --rm -v $(pwd)/outputs:/workspace/outputs edo-tts
```

## Option 2: Python 3.11 Virtual Environment

### Step 1: Install Python 3.11
```bash
# Using pyenv (recommended)
brew install pyenv
pyenv install 3.11.8
pyenv local 3.11.8

# Or using conda
conda create -n edo-tts python=3.11
conda activate edo-tts
```

### Step 2: Install Dependencies
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install TTS==0.22.0
pip install tensorboard pandas numpy matplotlib
```

### Step 3: Train Model
```bash
python -m TTS.bin.train_tts --config_path edo_config.json
```

## Option 3: Cloud Training (Google Colab)

### Step 1: Upload to Google Drive
1. Zip your edo-tts-model folder
2. Upload to Google Drive
3. Open Google Colab

### Step 2: Colab Training Notebook
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install TTS
!pip install TTS torch torchaudio

# Extract dataset
!unzip /content/drive/MyDrive/edo-tts-model.zip
%cd edo-tts-model

# Start training
!python -m TTS.bin.train_tts --config_path edo_config.json
```

## Option 4: Alternative TTS Framework (Piper)

If Coqui TTS continues to have issues, try Piper TTS:

### Step 1: Install Piper
```bash
pip install piper-tts
```

### Step 2: Convert Dataset Format
```python
# convert_to_piper.py
import json
import pandas as pd

df = pd.read_csv('metadata.csv', sep='|', header=None, names=['audio', 'text'])

# Create Piper format
piper_data = []
for _, row in df.iterrows():
    piper_data.append({
        "audio_path": row['audio'],
        "text": row['text']
    })

with open('piper_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(piper_data, f, indent=2, ensure_ascii=False)
```

### Step 3: Train with Piper
```bash
piper-train --dataset piper_dataset.json --output-dir outputs_piper
```

## Training Configuration Details

Your current configuration (`edo_config.json`):

```json
{
  "model_name": "edo_tts_tacotron2",
  "language": "edo",
  "characters": " '-abdefghiklmnoprstuvwyzàáèéìíòóùú̀ẹọ",
  "sample_rate": 22050,
  "batch_size": 16,
  "epochs": 1000,
  "learning_rate": 0.001
}
```

### Optimizations for Small Dataset:
- **Batch size**: 8-16 (adjust based on memory)
- **Learning rate**: 0.0005 (lower for stability)
- **Epochs**: 2000+ (more iterations needed)
- **Data augmentation**: Consider speed/pitch variations

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=outputs/
# Open http://localhost:6006
```

### Key Metrics to Watch:
1. **Loss curves** (training & validation)
2. **Attention alignment plots** (should be diagonal)
3. **Audio samples** (quality improves over time)
4. **Learning rate schedule**

## Expected Results

### Training Timeline:
- **Hours 1-2**: Basic phoneme alignment
- **Hours 3-6**: Improved pronunciation
- **Hours 8-12**: Natural prosody
- **Hours 12+**: Production quality

### Quality Milestones:
- **Epoch 100**: Words recognizable
- **Epoch 500**: Sentences clear
- **Epoch 1000**: Natural speech
- **Epoch 2000**: Production ready

## Testing Your Model

After training completes:

```python
# test_model.py
from TTS.api import TTS

# Load your trained model
tts = TTS(model_path="outputs/best_model.pth", 
          config_path="outputs/config.json")

# Test with Edo text
test_sentences = [
    "ẹ̀dó",
    "amẹ odidọn", 
    "mwaan ọnrẹn ne a rẹn vbene ọ tan sẹ hẹẹ"
]

for i, text in enumerate(test_sentences):
    tts.tts_to_file(text=text, file_path=f"test_{i}.wav")
    print(f"Generated: test_{i}.wav")
```

## Deployment Options

### 1. Local API Server
```python
# server.py
from TTS.api import TTS
from flask import Flask, request, send_file
import tempfile

app = Flask(__name__)
tts = TTS(model_path="outputs/best_model.pth")

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.json['text']
    with tempfile.NamedTemporaryFile(suffix='.wav') as f:
        tts.tts_to_file(text=text, file_path=f.name)
        return send_file(f.name, as_attachment=True)

app.run(host='0.0.0.0', port=5000)
```

### 2. Web Interface
```html
<!-- index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Edo TTS</title>
</head>
<body>
    <h1>Edo Language Text-to-Speech</h1>
    <textarea id="text" placeholder="Enter Edo text..."></textarea>
    <button onclick="synthesize()">Generate Speech</button>
    <audio id="audio" controls></audio>
    
    <script>
        function synthesize() {
            const text = document.getElementById('text').value;
            fetch('/synthesize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            })
            .then(response => response.blob())
            .then(blob => {
                const url = URL.createObjectURL(blob);
                document.getElementById('audio').src = url;
            });
        }
    </script>
</body>
</html>
```

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch_size to 4 or 8
2. **Poor Attention**: Increase epochs, check text cleaning
3. **Robotic Voice**: More training epochs needed
4. **Slow Training**: Use GPU, reduce audio length

### Performance Tips:

1. **GPU Training**: Much faster than CPU
2. **Mixed Precision**: Enable for memory efficiency  
3. **Data Quality**: Clean audio is crucial
4. **Batch Size**: Balance between memory and speed

## Next Steps

1. **Choose training method** (Docker recommended)
2. **Start training** (will take 6-12 hours)
3. **Monitor progress** with TensorBoard
4. **Test model** quality every few hours
5. **Deploy** when satisfied with quality

Your dataset is well-prepared and ready for training. The key is patience - TTS models need time to learn the nuances of human speech, especially for less common languages like Edo.

Good luck with your Edo TTS model training! 🚀
