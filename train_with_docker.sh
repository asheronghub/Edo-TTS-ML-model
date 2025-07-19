#!/bin/bash

# Edo TTS Training with Docker
# This script handles the complete training process using Docker

echo "🚀 Edo TTS Training Setup with Docker"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first:"
    echo "   Visit: https://docker.com/get-started"
    exit 1
fi

echo "✅ Docker found"

# Check if we have the required files
REQUIRED_FILES=("edo_config.json" "metadata.csv" "Dockerfile")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing file: $file"
        echo "   Please run simple_setup.py first"
        exit 1
    fi
done

echo "✅ All required files found"

# Count audio files
WAV_COUNT=$(find wavs/ -name "*.wav" 2>/dev/null | wc -l)
echo "🎵 Audio files: $WAV_COUNT"

if [ "$WAV_COUNT" -lt 100 ]; then
    echo "⚠️  Warning: Low number of audio files. Consider adding more for better quality."
fi

# Build Docker image
echo ""
echo "🏗️  Building Docker image..."
docker build -t edo-tts-trainer .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully"

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Ask user about GPU support
echo ""
read -p "Do you have NVIDIA GPU support? (y/N): " use_gpu

if [[ $use_gpu =~ ^[Yy]$ ]]; then
    DOCKER_ARGS="--gpus all"
    echo "🔥 Using GPU acceleration"
else
    DOCKER_ARGS=""
    echo "💻 Using CPU (training will be slower)"
fi

# Start training
echo ""
echo "🏃‍♂️ Starting training..."
echo "📈 You can monitor progress in real-time by opening another terminal and running:"
echo "   docker logs -f \$(docker ps --format '{{.Names}}' | grep edo-tts)"
echo ""
echo "⏰ Training will take several hours. Be patient!"
echo "🛑 To stop training: Ctrl+C"
echo ""

# Run training container
docker run --rm $DOCKER_ARGS \
    -v "$(pwd)/outputs:/workspace/outputs" \
    -v "$(pwd)/wavs:/workspace/wavs" \
    --name edo-tts-training \
    edo-tts-trainer

echo ""
echo "🎉 Training completed!"
echo "📁 Check outputs/ directory for your trained model"
echo "🎯 To test your model, look for best_model.pth in outputs/"
