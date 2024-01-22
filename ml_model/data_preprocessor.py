import glob
import os

from data_preparation.audio_processor import AudioPreprocessor
from mci_lib_ml.data_balancer import DataBalancer
from ml_model.audio_dataset import AudioDataSet
from mci_lib_ml.ml_data_set import MLDataSet


class DataPreprocessor:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

        self.RAW_AUDIO_DIRECTORY = r'C:\Users\User\Documents\Sound recordings'
        self.SEGMENTED_AUDIO_DIRECTORY = r'C:\Users\User\Documents\Sound recordings\segmented'
        self.PROCESSED_AUDIO_DIRECTORY = r'C:\Users\User\Documents\Sound recordings\processed'
        self.DATASET_SAVE_FILENAME = 'voice_dataset.cache'
        self.AUDIO_SAVE_FILE_LABELS = ['actor', 'size', 'weight', 'sound', 'sound detail', 'sound source', 'index',
                                       'augmentations']
        self.audio_processor = AudioPreprocessor(sample_rate=44100)

        self.inference_column_shapes = {
            'audio': (20032,),
        }

    def load_dataset_from_directory(self):
        ml_dataset = self.create_table_from_data_dir(
            self.RAW_AUDIO_DIRECTORY,
            ['actor', 'size', 'weight', 'sound', 'sound detail', 'sound source'])

        # Convert MLDataSet to AudioDataSet
        dataset = AudioDataSet()
        dataset.data = ml_dataset.data  # Transfer data to the AudioDataSet

        # Load the audio files into the dataset as binary audio
        dataset.load_audio_to_column()

        return dataset

    def preprocess_for_training(self, dataset):
        self.standardize_dataset(dataset, skip_segmentation=False)
        self.augment_dataset(dataset)
        self.extract_features(dataset)
        self.save_preprocessed_audio_to_files(dataset, self.SEGMENTED_AUDIO_DIRECTORY)
        self.normalize_and_encode_data(dataset)
        self.validate_preprocessed_dataset(dataset)

        dataset = DataBalancer.balance(dataset,
                                       filter_columns=[],
                                       balance_columns=['actor', 'size', 'weight', 'sound', 'sound detail'])

        return dataset

    def preprocess_for_inference(self, audio):
        dataset = AudioDataSet()
        dataset.add_sample({'audio': audio})
        self.standardize_dataset(dataset, skip_segmentation=True)
        dataset.truncate_column_to_shape('audio', self.inference_column_shapes['audio'])
        self.extract_features(dataset)
        self.normalize_and_encode_data(dataset)
        self.validate_preprocessed_dataset(dataset)
        column = dataset.get_columns(['spectrogram'])
        array = column.to_numpy_arrays()
        return array

    def standardize_dataset(self, dataset, skip_segmentation=False):
        if self.hyperparameters['normalize_volume_and_amplitude']:
            dataset.normalize_volume_and_amplitude()
        if not skip_segmentation:
            dataset.segment_audio(segment_length=0.5, discard_last_segment=True)

    def augment_dataset(self, dataset):
        # Augment the data to improve generalization and increase the number of training samples
        dataset.randomize_pitch_and_speed(pitch_factor_min=0.9, pitch_factor_max=1.1,
                                          speed_factor_min=0.9, speed_factor_max=1.1)
        dataset.augment_with_noise(noise_level=0.005)

        # Ensure a consistent length of audio before extracting MFCC
        # and spectrogram features. Altering the speed of the audio
        # changes its length.
        dataset.fix_column_shape_by_truncation('audio')
        dataset.truncate_column_to_divisible_shape('audio', 64)

    def extract_features(self, dataset):
        if self.hyperparameters['use_mfcc']:
            dataset.extract_mfcc_features(mfcc_column_name='mfcc', n_mfcc=self.hyperparameters['mfcc_coefficients'])
        if self.hyperparameters['use_spectrogram']:
            dataset.extract_spectrogram_features(spectrogram_column_name='spectrogram')
        if self.hyperparameters['use_hnr_mean_and_std']:
            dataset.extract_hnr_mean_and_std(new_column_name='hnr_mean_and_std')
        if self.hyperparameters['use_hnr_over_time']:
            dataset.extract_hnr_over_time(new_column_name='hnr_over_time')
        if self.hyperparameters['use_spectral']:
            dataset.extract_spectral_features(new_column_name='spectral')
        if self.hyperparameters['use_wavelet_at_level']:
            dataset.extract_wavelet_at_level('wavelet_at_level', level=self.hyperparameters['wavelet_level'])
        if self.hyperparameters['use_wavelets_mean_and_std']:
            dataset.extract_wavelets_mean_and_std(new_column_name='wavelets_mean_and_std')

    def save_preprocessed_audio_to_files(self, dataset, directory):
        for sample in dataset.data:
            filename = self.sample_to_filename(sample, self.AUDIO_SAVE_FILE_LABELS, '.wav')
            file_path = os.path.join(directory, filename)

            # Assuming 'audio' key in each record contains the audio segment
            audio_data = sample['audio']

            # Save the audio data to a file
            self.audio_processor.save_audio(file_path, audio_data)

    def normalize_and_encode_data(self, dataset):
        # Normalize the mean and variance of input features
        if self.hyperparameters['use_raw_audio']:
            dataset.normalize_mean_and_variance('audio')
        if self.hyperparameters['use_mfcc']:
            dataset.normalize_mean_and_variance('mfcc')
        if self.hyperparameters['use_spectrogram']:
            dataset.normalize_mean_and_variance('spectrogram')
        if self.hyperparameters['use_hnr_mean_and_std']:
            pass
        if self.hyperparameters['use_hnr_over_time']:
            dataset.normalize_mean_and_variance('hnr_over_time')
        if self.hyperparameters['use_spectral']:
            pass
        if self.hyperparameters['use_wavelet_at_level']:
            dataset.normalize_mean_and_variance('wavelet_at_level')
        if self.hyperparameters['use_wavelets_mean_and_std']:
            pass

        # Encode categorical features to numerical values
        label_encodings = dataset.encode_categorical_columns(['size', 'weight'])

        return label_encodings

    def validate_preprocessed_dataset(self, dataset):
        feature_columns = self.get_feature_columns()

        dataset.validate_multiple_column_shapes(feature_columns)
        # TODO: Ensure no data is nan, inf, or out-of-range

    def get_feature_columns(self):
        feature_columns = []
        if self.hyperparameters['use_raw_audio']:
            feature_columns.append('audio')
        if self.hyperparameters['use_mfcc']:
            feature_columns.append('mfcc')
        if self.hyperparameters['use_spectrogram']:
            feature_columns.append('spectrogram')
        if self.hyperparameters['use_hnr_mean_and_std']:
            feature_columns.append('hnr_mean_and_std')
        if self.hyperparameters['use_hnr_over_time']:
            feature_columns.append('hnr_over_time')
        if self.hyperparameters['use_spectral']:
            feature_columns.append('spectral')
        if self.hyperparameters['use_wavelet_at_level']:
            feature_columns.append('wavelet_at_level')
        if self.hyperparameters['use_wavelets_mean_and_std']:
            feature_columns.append('wavelets_mean_and_std')
        return feature_columns

    def create_table_from_data_dir(self, directory_path, column_titles, file_extensions=['wav', 'm4a', 'mp3']):
        ml_dataset = MLDataSet()

        # Combine file paths from both wav and m4a extensions
        file_paths = []
        for ext in file_extensions:
            file_paths.extend(glob.glob(os.path.join(directory_path, f'*.{ext}')))

        # Iterate over combined file paths
        for filepath in file_paths:
            sample = self.filename_to_labels(filepath, column_titles)
            sample['file_path'] = filepath

            # Add the record to the MLDataSet
            ml_dataset.add_sample(sample)

        return ml_dataset

    def save_audio_column_to_files(self, dataset: MLDataSet, column_name, directory_path: str, column_titles):
        for sample in dataset.data:
            filename = self.sample_to_filename(sample, column_titles, '.wav')
            file_path = os.path.join(directory_path, filename)

            # Assuming 'audio' key in each record contains the audio segment
            audio_data = sample[column_name]

            # Save the audio data to a file
            self.audio_processor.save_audio(file_path, audio_data)

    def save_binary_column_to_files(self, dataset: MLDataSet, column_name, directory_path: str):
        for sample in dataset.data:
            filename = self.sample_to_filename(sample, '.bin')
            file_path = os.path.join(directory_path, filename)

            data = sample[column_name]

            # Save the data to a file
            with open(file_path, "wb") as file_out:
                file_out.write(data)

    def sample_to_filename(self, sample, column_titles, file_extension=''):
        # Select labels from the sample based on column_titles
        labels = []
        for col in column_titles:
            if col in sample:
                if isinstance(sample[col], list):
                    # Join list items with plus signs
                    labels.append('+'.join(map(str, sample[col])))
                else:
                    # Append non-list items directly
                    labels.append(str(sample[col]))
        filename = self.labels_to_filename(labels, file_extension)
        return filename

    def labels_to_filename(self, labels, file_extension=""):
        filename = '_'.join(labels) + file_extension
        return filename

    def filename_to_labels(self, filename, column_headers):
        combined_labels = os.path.basename(filename).split('.')[0]  # Remove the file extension
        unstructured_labels = combined_labels.split('_')
        labels = {column: value for column, value in zip(column_headers, unstructured_labels)}
        return labels
