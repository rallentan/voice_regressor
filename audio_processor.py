import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def load_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio

    def save_audio(self, file_path, audio_data):
        # Write the audio data to the file in WAV format
        sf.write(file_path, audio_data, self.sample_rate)

    def normalize_volume_and_amplitude(self, audio):
        return librosa.util.normalize(audio)

    def segment_audio(self, audio, segment_length):
        # Assuming segment_length is in seconds
        segment_length_samples = int(segment_length * self.sample_rate)
        return [audio[i:i+segment_length_samples] for i in range(0, len(audio), segment_length_samples)]

    def change_pitch_and_speed(self, audio, pitch_factor, speed_factor):
        audio_pitch_shifted = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=pitch_factor)
        audio_speed_changed = librosa.effects.time_stretch(audio_pitch_shifted, rate=speed_factor)
        return audio_speed_changed

    def add_noise(self, audio, noise_level=0.005):
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_level * noise
        return np.clip(augmented_audio, -1.0, 1.0)

    def normalize_mean_and_variance(self, audio):
        return (audio - np.mean(audio)) / np.std(audio)

    def extract_mfcc_features(self, audio):
        return librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)

    def generate_spectrogram(self, audio, show=False):
        X = librosa.stft(audio)
        Xdb = librosa.amplitude_to_db(abs(X))
        if show:
            plt.figure(figsize=(14, 5))
            librosa.display.specshow(Xdb, sr=self.sample_rate, x_axis='time', y_axis='hz')
            plt.colorbar()
        return Xdb

# Example usage
# preprocessor = AudioPreprocessor()
# audio = preprocessor.load_audio('path_to_audio_file.wav')
# audio = preprocessor.normalize_volume_and_amplitude(audio)
# segments = preprocessor.segment_audio(audio, segment_length=5)  # 5 seconds segments
# pitch_speed_modified_audio = [preprocessor.change_pitch_and_speed(seg, pitch_factor=1.0, speed_factor=1.0) for seg in segments]
# noisy_audio = [preprocessor.add_noise(seg) for seg in segments]
# normalized_audio = [preprocessor.normalize_mean_and_variance(seg) for seg in segments]
# mfcc_features = [preprocessor.extract_mfcc_features(seg) for seg in segments]
# spectrograms = [preprocessor.generate_spectrogram(seg) for seg in segments]
