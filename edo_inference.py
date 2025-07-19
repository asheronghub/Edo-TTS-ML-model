#!/usr/bin/env python3
"""
Edo TTS Inference Script
Test the trained model on new Edo text
"""

import torch
import torch.nn as nn
import torchaudio
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf

class EdoTTSTransformer(nn.Module):
    """Transformer-based TTS model for Edo language - same architecture as training"""
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, n_mels=80):
        super(EdoTTSTransformer, self).__init__()
        
        self.d_model = d_model
        self.n_mels = n_mels
        
        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))  # Max length 1000
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Duration predictor
        self.duration_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
            nn.Softplus()
        )
        
        # Mel decoder
        self.mel_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_mels)
        )
        
        # Postnet for mel refinement
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, n_mels, 5, padding=2)
        )
        
    def forward(self, texts, text_lengths):
        batch_size, seq_len = texts.shape
        
        # Text embedding with positional encoding
        embedded = self.text_embedding(texts) * (self.d_model ** 0.5)
        embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Create padding mask
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=texts.device)
        for i, length in enumerate(text_lengths):
            padding_mask[i, length:] = True
        
        # Transformer encoding
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        # Duration prediction
        durations = self.duration_predictor(encoded).squeeze(-1)
        
        # Length regulation (expand encoded features)
        expanded_features = encoded.repeat_interleave(2, dim=1)  # Simple 2x expansion
        
        # Mel prediction
        mel_output = self.mel_decoder(expanded_features)
        
        # Postnet refinement
        mel_postnet = mel_output.transpose(1, 2)
        mel_refined = self.postnet(mel_postnet)
        mel_output_refined = mel_output + mel_refined.transpose(1, 2)
        
        return {
            'mel_output': mel_output,
            'mel_output_refined': mel_output_refined,
            'durations': durations,
            'encoded': encoded
        }

class EdoTTSInference:
    """Inference class for Edo TTS"""
    
    def __init__(self, model_path, model_info_path, device=None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Load model info
        with open(model_info_path, 'r', encoding='utf-8') as f:
            self.model_info = json.load(f)
        
        self.char_to_idx = self.model_info['char_to_idx']
        self.idx_to_char = {int(idx): char for char, idx in self.char_to_idx.items()}
        self.vocab_size = self.model_info['vocab_size']
        
        # Create model
        self.model = EdoTTSTransformer(vocab_size=self.vocab_size)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded Edo TTS model from {model_path}")
        print(f"🔧 Using device: {self.device}")
        print(f"📝 Vocabulary size: {self.vocab_size}")
        
    def text_to_indices(self, text):
        """Convert text to model input indices"""
        indices = []
        for char in text.lower():
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                print(f"⚠️  Unknown character '{char}', skipping")
        return indices
    
    def synthesize(self, text, output_path="output.wav"):
        """Synthesize speech from Edo text"""
        
        print(f"🎯 Synthesizing: '{text}'")
        
        # Convert text to indices
        text_indices = self.text_to_indices(text)
        if not text_indices:
            print("❌ No valid characters found in text!")
            return None
        
        # Prepare input
        text_tensor = torch.tensor([text_indices], dtype=torch.long).to(self.device)
        text_length = torch.tensor([len(text_indices)]).to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            outputs = self.model(text_tensor, text_length)
            mel_output = outputs['mel_output_refined'][0].cpu().numpy()  # Remove batch dimension
        
        print(f"🎵 Generated mel spectrogram: {mel_output.shape}")
        
        # Convert mel to audio (simplified - would need vocoder for real synthesis)
        # For now, we'll create a visualization and save the mel spectrogram
        
        # Visualize mel spectrogram
        plt.figure(figsize=(12, 6))
        plt.imshow(mel_output.T, aspect='auto', origin='lower', interpolation='nearest')
        plt.title(f"Edo TTS Generated Mel Spectrogram: '{text}'")
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.colorbar()
        
        mel_plot_path = output_path.replace('.wav', '_mel.png')
        plt.savefig(mel_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved mel spectrogram visualization: {mel_plot_path}")
        
        # Save mel spectrogram data
        mel_data_path = output_path.replace('.wav', '_mel.npy')
        np.save(mel_data_path, mel_output)
        print(f"💾 Saved mel spectrogram data: {mel_data_path}")
        
        # Simple audio generation (placeholder - would need proper vocoder)
        # Generate simple sine wave approximation based on mel
        sample_rate = 22050
        duration = mel_output.shape[0] * 256 / sample_rate  # hop_length=256
        
        # Create a simple audio signal (demonstration only)
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Use average mel frequency as fundamental frequency (very simplified)
        freq = 200 + np.mean(mel_output) * 2  # Rough frequency mapping
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        # Add some variation based on mel
        for i in range(min(5, mel_output.shape[1])):
            mel_freq_profile = np.interp(t, np.linspace(0, duration, mel_output.shape[0]), mel_output[:, i])
            audio += 0.1 * np.sin(2 * np.pi * (freq + i * 50) * t + mel_freq_profile * 0.1)
        
        # Save audio
        sf.write(output_path, audio, sample_rate)
        print(f"🎵 Saved synthesized audio: {output_path}")
        
        return {
            'mel_spectrogram': mel_output,
            'audio': audio,
            'mel_plot_path': mel_plot_path,
            'mel_data_path': mel_data_path,
            'audio_path': output_path
        }

def test_edo_tts():
    """Test the Edo TTS model with sample texts"""
    
    # Check for model files
    model_files = list(Path(".").glob("edo_tts_best_epoch_*.pth"))
    if not model_files:
        print("❌ No trained model found!")
        print("🔧 Training should create edo_tts_best_epoch_X.pth files")
        return
    
    # Use the latest model
    latest_model = sorted(model_files)[-1]
    print(f"🤖 Using model: {latest_model}")
    
    if not Path("edo_model_info.json").exists():
        print("❌ Model info file not found!")
        return
    
    # Create inference engine
    tts = EdoTTSInference(latest_model, "edo_model_info.json")
    
    # Test sentences in Edo
    test_sentences = [
        "ovbe",           # Simple word
        "ọse",            # Word with tone mark
        "oberhọmwan",     # Longer word
        "sikẹ odaro",     # Two words
        "imina oghọghọ"   # Sentence with tone marks
    ]
    
    print("\n🎯 Testing Edo TTS Synthesis:")
    print("=" * 50)
    
    for i, sentence in enumerate(test_sentences):
        print(f"\n📝 Test {i+1}: '{sentence}'")
        try:
            result = tts.synthesize(sentence, f"edo_synthesis_test_{i+1}.wav")
            if result:
                print(f"✅ Successfully generated audio for '{sentence}'")
                print(f"   📊 Mel shape: {result['mel_spectrogram'].shape}")
                print(f"   🎵 Audio length: {len(result['audio'])/22050:.2f} seconds")
            else:
                print(f"❌ Failed to generate audio for '{sentence}'")
        except Exception as e:
            print(f"❌ Error synthesizing '{sentence}': {e}")
    
    print("\n🎉 Edo TTS Testing Complete!")
    print("\n📁 Generated files:")
    for file in Path(".").glob("edo_synthesis_test_*"):
        print(f"  📄 {file}")

def main():
    """Main function"""
    print("🎯 Edo TTS Inference Tool")
    print("=" * 40)
    
    # Test with sample sentences
    test_edo_tts()
    
    print("\n💡 Usage:")
    print("  🐍 python edo_inference.py")
    print("  🎯 This will test the model with sample Edo phrases")
    print("  📊 Check the generated mel spectrograms and audio files")
    
    print("\n⚠️  Note:")
    print("  This is a basic synthesis demo. For high-quality audio,")
    print("  you would need to add a vocoder (e.g., HiFi-GAN) to")
    print("  convert mel spectrograms to realistic speech audio.")

if __name__ == "__main__":
    main()
