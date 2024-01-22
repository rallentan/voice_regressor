import random

from data_preparation.audio_processor import AudioPreprocessor
from mci_lib_ml.ml_data_set import MLDataSet


class AudioDataSet(MLDataSet):
    def __init__(self):
        super().__init__()
        self.audio_processor = AudioPreprocessor(sample_rate=44100)

    def load_audio_to_column(self):
        self.extract_new_column('audio', lambda sample: self.audio_processor.load_audio(sample['file_path']))

    def normalize_volume_and_amplitude(self):
        self.transform_column('audio',
                              lambda audio: self.audio_processor.normalize_volume_and_amplitude(audio))

    def randomize_pitch_and_speed(self, pitch_factor_min=1.0, pitch_factor_max=1.0, speed_factor_min=1.0, speed_factor_max=1.0):
        self.transform_column('audio',
                              lambda audio: self.audio_processor.change_pitch_and_speed(
                                  audio,
                                  random.uniform(pitch_factor_min, pitch_factor_max),
                                  random.uniform(speed_factor_min, speed_factor_max)))

    def augment_pitch_and_speed(self, pitch_factor_min=1.0, pitch_factor_max=1.0, speed_factor_min=1.0, speed_factor_max=1.0):
        self.augment_column('audio',
                            lambda sample: self.audio_processor.change_pitch_and_speed(
                                sample['audio'],
                                random.uniform(pitch_factor_min, pitch_factor_max),
                                random.uniform(speed_factor_min, speed_factor_max)))

    def add_noise(self, noise_level=0.005):
        self.transform_column('audio', lambda audio: self.audio_processor.add_noise(audio, noise_level=0.005))

    def augment_with_noise(self, noise_level=0.005):
        self.augment_column('audio',
                            'noise',
                            lambda sample: self.audio_processor.add_noise(sample['audio'], noise_level=0.005))

    def extract_mfcc_features(self, mfcc_column_name, n_mfcc=13):
        self.extract_new_column(mfcc_column_name,
                                lambda sample: self.audio_processor.extract_mfcc_features(sample['audio'], n_mfcc))

    def extract_spectrogram_features(self, spectrogram_column_name):
        self.extract_new_column(spectrogram_column_name,
                                lambda sample: self.audio_processor.generate_spectrogram(sample['audio']))

    def extract_hnr_mean_and_std(self, new_column_name):
        self.extract_new_column(new_column_name,
                                lambda sample: self.audio_processor.extract_hnr_mean_and_std(sample['audio']))

    def extract_hnr_over_time(self, new_column_name):
        self.extract_new_column(new_column_name,
                                lambda sample: self.audio_processor.extract_hnr_over_time(sample['audio']))

    def extract_spectral_features(self, new_column_name):
        self.extract_new_column(new_column_name,
                                lambda sample: self.audio_processor.extract_spectral_features(sample['audio']))

    def extract_wavelet_at_level(self, new_column_name, level=5):
        self.extract_new_column(new_column_name,
                                lambda sample: self.audio_processor.extract_wavelet_at_level(sample['audio'], level))

    def extract_wavelets_mean_and_std(self, new_column_name):
        self.extract_new_column(new_column_name,
                                lambda sample: self.audio_processor.extract_wavelets_mean_and_std(sample['audio']))

    def cull_short_segments(self, length):
        """
        Remove audio samples that are shorter than the specified length.

        Args:
        length (int): The minimum required length of the audio samples.
        """
        self.remove(lambda sample: len(sample['audio']) < length)

    def segment_audio(self, segment_length=0.5, discard_last_segment=False):
        audio_processor = AudioPreprocessor()
        segmented_dataset = MLDataSet()

        for record in self.data:
            segments = audio_processor.segment_audio(record['audio'], segment_length)

            # Initialize a counter for each record
            segment_index = 0

            for seg in segments:
                # If discard_short_segments is True, skip the last segment
                if discard_last_segment and segment_index == len(segments) - 1:
                    break

                new_record = record.copy()
                new_record['audio'] = seg
                new_record['index'] = segment_index  # Add 'index' to the record
                segmented_dataset.add_sample(new_record)

                # Increment the segment index for the next segment
                segment_index += 1

        self.data = segmented_dataset.data
