FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies step by step to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core ML libraries
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install TTS dependencies 
RUN pip install --no-cache-dir \
    numpy scipy librosa soundfile \
    tensorboard pandas matplotlib \
    tqdm pyyaml inflect anyascii \
    coqpit trainer numba scikit-learn \
    bangla pysbd gruut phonemizer

# Install TTS (now with all dependencies available)
RUN pip install --no-cache-dir TTS==0.22.0

# Set working directory
WORKDIR /workspace

# Copy all files
COPY . .

# Create outputs directory
RUN mkdir -p outputs

# Default command
CMD ["python", "-m", "TTS.bin.train_tts", "--config_path", "edo_config.json"]
