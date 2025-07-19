#!/usr/bin/env python3
"""
Edo TTS with Griffin-Lim Vocoder
Convert mel-spectrograms to realistic speech audio
"""

import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import soundfile as sf
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

class GriffinLimVocoder:
    """Griffin-Lim vocoder for converting mel-spectrograms to audio"""
    
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80, power=1.2):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power
        
        # Mel-to-linear conversion
        self.mel_to_linear = T.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        
        # Griffin-Lim algorithm
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
            n_iter=50  # Iterations for phase reconstruction
        )
        
        print(f"🎵 Griffin-Lim Vocoder initialized")
        print(f"   📊 Sample rate: {sample_rate} Hz")
        print(f"   🔊 N_FFT: {n_fft}, Hop: {hop_length}")
        print(f"   🎯 Griffin-Lim iterations: 50")
    
    def mel_to_audio(self, mel_spectrogram):
        """Convert mel-spectrogram to audio using Griffin-Lim"""
        
        # Ensure tensor format
        if isinstance(mel_spectrogram, np.ndarray):
            mel_tensor = torch.from_numpy(mel_spectrogram).float()
        else:
            mel_tensor = mel_spectrogram.float()
        
        # Add batch dimension if needed
        if len(mel_tensor.shape) == 2:
            mel_tensor = mel_tensor.unsqueeze(0)  # (1, time, mels)
        
        # Transpose to (batch, mels, time) for torchaudio
        mel_tensor = mel_tensor.transpose(1, 2)
        
        print(f"🔧 Converting mel shape: {mel_tensor.shape}")
        
        # Convert mel to linear spectrogram
        linear_spec = self.mel_to_linear(mel_tensor)
        
        # Apply Griffin-Lim to reconstruct phase and convert to audio
        audio = self.griffin_lim(linear_spec)
        
        # Remove batch dimension
        audio = audio.squeeze(0).numpy()
        
        print(f"🎵 Generated audio shape: {audio.shape}")
        print(f"⏱️  Audio duration: {len(audio) / self.sample_rate:.2f} seconds")
        
        return audio

# Update the inference class to use the vocoder
class EdoTTSInferenceWithVocoder:
    """Enhanced Edo TTS inference with proper vocoder"""
    
    def __init__(self, model_path, model_info_path, device=None):
        # Import here to avoid circular imports
        from edo_inference import EdoTTSTransformer
        
        if device is None:
            self.device = torch.device("cpu")  # Use CPU for stability
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
        
        # Initialize vocoder
        self.vocoder = GriffinLimVocoder()
        
        print(f"✅ Loaded Edo TTS model with Griffin-Lim vocoder")
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
    
    def synthesize_realistic_speech(self, text, output_path="realistic_edo_speech.wav"):
        """Synthesize realistic speech from Edo text using vocoder"""
        
        print(f"🎯 Synthesizing realistic speech: '{text}'")
        
        # Convert text to indices
        text_indices = self.text_to_indices(text)
        if not text_indices:
            print("❌ No valid characters found in text!")
            return None
        
        print(f"🔤 Text indices: {text_indices}")
        
        # Prepare input
        text_tensor = torch.tensor([text_indices], dtype=torch.long).to(self.device)
        text_length = torch.tensor([len(text_indices)]).to(self.device)
        
        print(f"📊 Input tensor shape: {text_tensor.shape}")
        
        # Generate mel spectrogram
        with torch.no_grad():
            outputs = self.model(text_tensor, text_length)
            mel_output = outputs['mel_output_refined'][0].cpu()  # Remove batch dimension
        
        print(f"🎵 Generated mel spectrogram shape: {mel_output.shape}")
        
        # Apply some post-processing to the mel-spectrogram
        mel_np = mel_output.numpy()
        
        # Normalize mel spectrogram
        mel_np = np.maximum(mel_np, mel_np.max() - 8.0)  # Dynamic range compression
        mel_np = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8)
        mel_np = mel_np * 2.0 - 1.0  # Scale to [-1, 1] range
        
        print(f"📊 Mel spectrogram range: {mel_np.min():.3f} to {mel_np.max():.3f}")
        
        # Convert mel to audio using Griffin-Lim vocoder
        audio = self.vocoder.mel_to_audio(mel_np)
        
        # Post-process audio
        audio = np.clip(audio, -1.0, 1.0)  # Clip to prevent distortion
        audio = audio * 0.5  # Reduce volume
        
        # Apply gentle filtering to smooth the audio
        from scipy import signal
        # Simple lowpass filter to remove high-frequency artifacts
        b, a = signal.butter(4, 8000, fs=22050, btype='low')
        audio = signal.filtfilt(b, a, audio)
        
        # Save audio
        sf.write(output_path, audio, self.vocoder.sample_rate)
        print(f"🎵 Saved realistic speech: {output_path}")
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Plot mel spectrogram
        plt.subplot(2, 2, 1)
        plt.imshow(mel_np.T, aspect='auto', origin='lower', interpolation='nearest')
        plt.title(f"Mel Spectrogram: '{text}'")
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.colorbar()
        
        # Plot audio waveform
        plt.subplot(2, 2, 2)
        time_axis = np.linspace(0, len(audio) / self.vocoder.sample_rate, len(audio))
        plt.plot(time_axis, audio)
        plt.title('Generated Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Plot frequency spectrum
        plt.subplot(2, 2, 3)
        freqs, times, Sxx = signal.spectrogram(audio, self.vocoder.sample_rate)
        plt.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-10))
        plt.title('Audio Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
        
        # Plot comparison
        plt.subplot(2, 2, 4)
        plt.plot(mel_np[:, :10].mean(axis=1), label='Average Mel Energy')
        plt.title('Mel Energy Profile')
        plt.xlabel('Mel Frequency Bin')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        vis_path = output_path.replace('.wav', '_analysis.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved analysis: {vis_path}")
        
        return {
            'audio': audio,
            'mel_spectrogram': mel_np,
            'sample_rate': self.vocoder.sample_rate,
            'duration': len(audio) / self.vocoder.sample_rate,
            'audio_path': output_path,
            'analysis_path': vis_path
        }

def test_realistic_edo_tts():
    """Test the enhanced Edo TTS with vocoder"""
    
    # Check for model files
    model_files = list(Path(".").glob("edo_tts_best_epoch_*.pth"))
    if not model_files:
        print("❌ No trained model found!")
        return
    
    latest_model = sorted(model_files)[-1]
    print(f"🤖 Using model: {latest_model}")
    
    if not Path("edo_model_info.json").exists():
        print("❌ Model info file not found!")
        return
    
    # Create enhanced inference engine
    tts = EdoTTSInferenceWithVocoder(latest_model, "edo_model_info.json")
    
    # Test sentences
    test_sentences = [
        "ovbe",           # Simple word
        "ọse",            # Word with tone mark
        "oberhọmwan",     # Longer word
        "sikẹ odaro",     # Two words
    ]
    
    print("\n🎯 Testing Enhanced Edo TTS with Vocoder:")
    print("=" * 60)
    
    for i, sentence in enumerate(test_sentences):
        print(f"\n📝 Test {i+1}: '{sentence}'")
        try:
            output_file = f"realistic_edo_{i+1}_{sentence.replace(' ', '_')}.wav"
            result = tts.synthesize_realistic_speech(sentence, output_file)
            
            if result:
                print(f"✅ Generated realistic speech: '{sentence}'")
                print(f"🎵 Audio file: {result['audio_path']}")
                print(f"📊 Analysis: {result['analysis_path']}")
                print(f"⏱️  Duration: {result['duration']:.2f} seconds")
                print(f"🔊 Sample rate: {result['sample_rate']} Hz")
                
                print(f"\n🎧 Play with: afplay {result['audio_path']}")
            else:
                print(f"❌ Failed to generate speech for '{sentence}'")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Enhanced Edo TTS testing complete!")
    print(f"🔊 The audio files should now sound like realistic speech!")

if __name__ == "__main__":
    test_realistic_edo_tts()
