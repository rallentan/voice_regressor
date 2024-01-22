import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import parselmouth
import pywt

class AudioPreprocessor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def load_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        if sr != self.sample_rate:
            raise "SR check failed"
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

    def extract_mfcc_features(self, audio, n_mfcc=13):
        return librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)

    def generate_spectrogram(self, audio, show=False):
        X = librosa.stft(audio)
        Xdb = librosa.amplitude_to_db(abs(X))
        if show:
            plt.figure(figsize=(14, 5))
            librosa.display.specshow(Xdb, sr=self.sample_rate, x_axis='time', y_axis='hz')
            plt.colorbar()
        return Xdb

    def extract_harmonic_noise_ratio(self, audio):
        # Create a Parselmouth Sound object from the NumPy array
        sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)

        # Calculate Harmonic-to-Noise Ratio
        hnr = sound.to_harmonicity()

        return hnr

    def extract_hnr_mean_and_std(self, audio):
        hnr = self.extract_harmonic_noise_ratio(audio)

        # Calculate the mean and std HNR for a consistent shape for ML tasks
        mean_hnr = np.mean(hnr.values)
        std_hnr = np.std(hnr.values)

        return np.array([mean_hnr, std_hnr])

    def extract_hnr_over_time(self, audio):
        hnr = self.extract_harmonic_noise_ratio(audio)
        return hnr.values

    def extract_spectral_features(self, audio):
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)

        # Concatenate features along the second axis
        combined_features = np.concatenate([spectral_centroid, spectral_bandwidth], axis=1)

        return combined_features

    def extract_wavelet_transform(self, audio):
        # Wavelet Transform using PyWavelets
        coefficients = pywt.wavedec(audio, 'db1')  # Using Daubechies wavelet
        return coefficients

    def extract_wavelet_at_level(self, audio, level=5):
        coefficients = self.extract_wavelet_transform(audio)

        # Select coefficients at the specified level
        selected_coefficients = coefficients[level]

        return selected_coefficients

    def extract_wavelets_mean_and_std(self, audio):
        coefficients = self.extract_wavelet_transform(audio)

        # Aggregate coefficients (e.g., mean, std, etc.)
        mean = [np.mean(coeff) for coeff in coefficients]

        # Aggregate coefficients (e.g., mean, std, etc.)
        std = [np.std(coeff) for coeff in coefficients]

        # Combine mean and std into a single array
        mean_and_std = np.array(mean + std)

        return mean_and_std
