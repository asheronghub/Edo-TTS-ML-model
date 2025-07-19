#!/usr/bin/env python3
"""
Test Your Trained Edo TTS Model
Run this script after training completes to test your model
"""

import os
import torch
from pathlib import Path

def test_edo_model():
    """Test the trained Edo TTS model"""
    
    # Check if model files exist
    model_path = "outputs/best_model.pth"
    config_path = "outputs/config.json"
    
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        print("Expected location: outputs/best_model.pth")
        print("Make sure training completed successfully.")
        return False
        
    if not os.path.exists(config_path):
        print("❌ Config file not found!")
        print("Expected location: outputs/config.json")
        return False
    
    print("✅ Model files found!")
    print(f"📁 Model: {model_path}")
    print(f"📁 Config: {config_path}")
    
    # Check model size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📊 Model size: {model_size_mb:.1f} MB")
    
    try:
        # Try to load the model
        from TTS.api import TTS
        
        print("🧠 Loading trained model...")
        tts = TTS(model_path=model_path, config_path=config_path)
        
        print("✅ Model loaded successfully!")
        
        # Test with Edo sentences
        test_sentences = [
            "ẹ̀dó",                              # "Edo"
            "amẹ odidọn",                        # Common phrase  
            "ebaan wẹ miẹn mwẹn a",             # Longer sentence
            "mwaan ọnrẹn ne a rẹn vbene ọ tan sẹ hẹẹ"  # Complex sentence
        ]
        
        print("🎵 Generating test audio files...")
        
        for i, sentence in enumerate(test_sentences):
            output_file = f"edo_test_{i+1}.wav"
            
            try:
                tts.tts_to_file(text=sentence, file_path=output_file)
                print(f"  ✅ Generated: {output_file} - '{sentence}'")
                
            except Exception as e:
                print(f"  ❌ Failed to generate {output_file}: {e}")
        
        print("\n🎉 Testing complete!")
        print("🎧 Listen to the generated files:")
        for i in range(len(test_sentences)):
            print(f"   - edo_test_{i+1}.wav")
            
        return True
        
    except ImportError:
        print("❌ TTS library not found.")
        print("Install with: pip install TTS")
        return False
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("The model might not be trained yet or there was an error during training.")
        return False

def create_simple_api():
    """Create a simple API for your Edo TTS model"""
    
    api_code = '''#!/usr/bin/env python3
"""
Simple Edo TTS API Server
Run this after training to serve your model
"""

from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import tempfile
import os

app = Flask(__name__)

# Load your trained model
try:
    tts = TTS(model_path="outputs/best_model.pth", config_path="outputs/config.json")
    print("✅ Edo TTS Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    tts = None

@app.route('/')
def home():
    return """
    <h1>🎵 Edo TTS API</h1>
    <p>POST to /synthesize with JSON: {"text": "your edo text"}</p>
    <p>Example: curl -X POST -H "Content-Type: application/json" -d '{"text":"ẹ̀dó"}' http://localhost:5000/synthesize --output edo.wav</p>
    """

@app.route('/synthesize', methods=['POST'])
def synthesize():
    if not tts:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tts.tts_to_file(text=text, file_path=tmp.name)
            return send_file(tmp.name, as_attachment=True, download_name='edo_speech.wav')
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": tts is not None,
        "supported_language": "edo"
    })

if __name__ == '__main__':
    print("🚀 Starting Edo TTS API server...")
    print("📡 Access at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
    
    with open('edo_tts_api.py', 'w', encoding='utf-8') as f:
        f.write(api_code)
    
    print("✅ Created edo_tts_api.py - Simple API server for your model")

def package_model_for_sharing():
    """Package your model for easy sharing"""
    
    print("📦 Packaging your Edo TTS model...")
    
    required_files = [
        "outputs/best_model.pth",
        "outputs/config.json", 
        "outputs/scale_stats.npy"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    # Create package script
    package_script = '''#!/bin/bash
# Edo TTS Model Package Creator

echo "📦 Creating Edo TTS model package..."

# Create package directory
mkdir -p edo_tts_package
cp outputs/best_model.pth edo_tts_package/
cp outputs/config.json edo_tts_package/
cp outputs/scale_stats.npy edo_tts_package/

# Create usage instructions
cat > edo_tts_package/README.md << 'EOF'
# Edo TTS Model Package

## Installation
```bash
pip install TTS torch torchaudio
```

## Usage
```python
from TTS.api import TTS

# Load the Edo TTS model
tts = TTS(model_path="best_model.pth", config_path="config.json")

# Generate speech
tts.tts_to_file(text="ẹ̀dó", file_path="edo_speech.wav")
```

## Supported Characters
Space, apostrophe, hyphen, and letters: abdefghiklmnoprstuvwyzàáèéìíòóùú̀ẹọ

## Language
This model was trained specifically for the Edo language of Nigeria.
EOF

# Create example usage script
cat > edo_tts_package/example.py << 'EOF'
#!/usr/bin/env python3
from TTS.api import TTS

# Load model
tts = TTS(model_path="best_model.pth", config_path="config.json")

# Test sentences
sentences = [
    "ẹ̀dó",
    "amẹ odidọn",
    "ebaan wẹ miẹn mwẹn a"
]

for i, text in enumerate(sentences):
    output_file = f"edo_example_{i+1}.wav"
    tts.tts_to_file(text=text, file_path=output_file)
    print(f"Generated: {output_file}")
EOF

# Create archive
tar -czf edo_tts_model_package.tar.gz edo_tts_package/

echo "✅ Package created: edo_tts_model_package.tar.gz"
echo "📁 Extract with: tar -xzf edo_tts_model_package.tar.gz"
'''
    
    with open('create_package.sh', 'w') as f:
        f.write(package_script)
    
    os.chmod('create_package.sh', 0o755)
    print("✅ Created create_package.sh - Run this to package your model for sharing")

if __name__ == "__main__":
    print("🎯 Edo TTS Model Testing & Deployment")
    print("=====================================")
    
    # Test the model
    success = test_edo_model()
    
    if success:
        print("\n🚀 Creating deployment scripts...")
        create_simple_api()
        package_model_for_sharing()
        
        print("\n🎉 All set! Your Edo TTS model is ready!")
        print("\n📋 Next steps:")
        print("1. Test audio files: edo_test_*.wav")
        print("2. Run API server: python edo_tts_api.py")
        print("3. Package for sharing: ./create_package.sh")
    else:
        print("\n⏳ Model not ready yet. Wait for training to complete.")
