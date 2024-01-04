import glob
import os

from conv_model import VoiceModel
from data_loader import DataLoader
from data_preparer import DataPreparer
from ml_data_set import MLDataSet
from audio_processor import AudioPreprocessor

raw_audio_directory = r'C:\Users\User\Documents\Sound recordings'
segmented_audio_directory = r'C:\Users\User\Documents\Sound recordings\segmented'
processed_audio_directory = r'C:\Users\User\Documents\Sound recordings\processed'

all_actors = ['actor-1',
              'actor-2',
              'actor-3',
              'actor-4',
              'actor-5',
              'actor-6',
              'actor-7',
              'actor-8',
              'actor-9',
              'actor-10']

all_labels = ['small-size',
              'large-size',
              'light-weight',
              'heavy-weight'] + all_actors


def main():
    dataset = create_table_from_data_dir(
        raw_audio_directory,
        ['actor', 'size', 'weight', 'sound', 'sound detail'])

    # Preprocess the dataset
    dataset = load_audio_to_column(dataset)
    dataset = normalize_volume_and_amplitude(dataset)
    dataset = segment_audio(dataset)
    dataset = change_pitch_and_speed(dataset)
    dataset = add_noise(dataset)

    # Save prepared data to disk for faster loading in future runs
    save_dataset_to_files(
        dataset,
        segmented_audio_directory,
        ['actor', 'size', 'weight', 'sound', 'sound detail', 'index'])

    # Extract features and normalize
    dataset = normalize_mean_and_variance(dataset)
    dataset = extract_mfcc_dataset(dataset)

    # Balance the data
    dataset = DataPreparer.balance(dataset,
                                   filter_columns=[],
                                   balance_columns=['actor', 'size', 'weight', 'sound', 'sound detail'])

    # Convert string labels to nominal values usable in keras
    label_encodings = dataset.encode_categorical_columns(['size', 'weight'])

    # Cull audio segments that are too short (i.e. the last segment of each recording)
    feature_length = len(dataset.data[0]['audio'])  # should be 11025 for 0.5s segments
    dataset.remove(lambda sample: len(sample['audio']) < feature_length)

    # Split the dataset into training, validation, and test sets
    train, val, test = dataset.split(train_size=0.7, val_size=0.15, test_size=0.15)

    # Get features and labels for training
    train_features = train.get_single_column('audio')
    train_labels = train.get_multiple_columns(['size', 'weight'])
    val_features = val.get_single_column('audio')
    val_labels = val.get_multiple_columns(['size', 'weight'])
    test_features = test.get_single_column('audio')
    test_labels = test.get_multiple_columns(['size', 'weight'])

    # Check shapes
    print("Shape (features/labels): ", train_features.shape, train_labels.shape)

    # Initialize the voice model (assuming feature_length is defined correctly)
    model = VoiceModel(feature_length)

    # Train the model
    history = model.train(train_features, train_labels, val_features, val_labels, epochs=10, batch_size=32)

    # Plot graphs
    model.plot_loss_over_epochs(history)

    # TODO: The plot falls off too abruptly
    # TODO: Get more training data
    # TODO: Generate graphs of model performance
    # TODO: Feed model real-time user input
    # TODO: Generate graph of model outputs

    # Evaluate the model on test data
    mse, mae = model.evaluate(test_features, test_labels)
    print(f"Mean Squared Error (MSE) on Test Data: {mse}")
    print(f"Mean Absolute Error (MAE) on Test Data: {mae}")

    # Optionally, save the trained model
    model.save('trained_voice_model.keras')


def create_table_from_data_dir(directory_path, column_titles, file_extensions=['wav', 'm4a']):
    ml_dataset = MLDataSet()

    # Combine file paths from both wav and m4a extensions
    file_paths = []
    for ext in file_extensions:
        file_paths.extend(glob.glob(os.path.join(directory_path, f'*.{ext}')))

    # Iterate over combined file paths
    for filepath in file_paths:
        filename = os.path.basename(filepath).split('.')[0]  # Remove the file extension

        # Extract values based on the filename and map them to the corresponding column titles
        values = filename.split('_')
        record = {column: value for column, value in zip(column_titles, values)}
        record['file_path'] = filepath

        # Add the record to the MLDataSet
        ml_dataset.add_sample(record)

    return ml_dataset


def load_audio_to_column(dataset: MLDataSet):
    return extract_new_column(
        dataset,
        'audio',
        lambda sample, audio_processor: audio_processor.load_audio(sample['file_path']))


def normalize_volume_and_amplitude(dataset: MLDataSet):
    return transform_column(
        dataset,
        'audio',
        lambda audio, audio_processor: audio_processor.normalize_volume_and_amplitude(audio))


def segment_audio(dataset: MLDataSet, segment_length=0.5):
    audio_processor = AudioPreprocessor()
    segmented_dataset = MLDataSet()

    for record in dataset.data:
        segments = audio_processor.segment_audio(record['audio'], segment_length)

        # Initialize a counter for each record
        segment_index = 0

        for seg in segments:
            new_record = record.copy()
            new_record['audio'] = seg
            new_record['index'] = segment_index  # Add 'index' to the record
            segmented_dataset.add_sample(new_record)

            # Increment the segment index for the next segment
            segment_index += 1

    return segmented_dataset


def change_pitch_and_speed(dataset: MLDataSet, pitch_factor=1.0, speed_factor=1.0):
    return transform_column(
        dataset,
        'audio',
        lambda audio, audio_processor: audio_processor.change_pitch_and_speed(audio, pitch_factor, speed_factor))


def add_noise(dataset: MLDataSet):
    return transform_column(dataset, 'audio', lambda audio, audio_processor: audio_processor.add_noise(audio))


def normalize_mean_and_variance(dataset: MLDataSet):
    return transform_column(
        dataset,
        'audio',
        lambda audio, audio_processor: audio_processor.normalize_mean_and_variance(audio))


def extract_mfcc_dataset(dataset: MLDataSet):
    return extract_new_column(
        dataset,
        'mfcc',
        lambda sample, audio_processor: audio_processor.extract_mfcc_features(sample['audio']))

def cull_short_segments(dataset: MLDataSet, length):
    return transform_column(
        dataset,
        'audio',
        lambda audio, audio_processor: audio_processor.normalize_volume_and_amplitude(audio))

def transform_column(dataset: MLDataSet, column, transformation_function):
    transformed_dataset = MLDataSet()

    for record in dataset.data:
        transformed_record = record.copy()
        transformed_record[column] = transformation_function(record[column], AudioPreprocessor())
        transformed_dataset.add_sample(transformed_record)

    return transformed_dataset


def extract_new_column(dataset: MLDataSet, new_column, transformation_function):
    transformed_dataset = MLDataSet()

    for record in dataset.data:
        transformed_record = record.copy()
        transformed_record[new_column] = transformation_function(transformed_record, AudioPreprocessor())
        transformed_dataset.add_sample(transformed_record)

    return transformed_dataset


def save_dataset_to_files(dataset: MLDataSet, directory_path: str, columns: list):
    audio_processor = AudioPreprocessor()

    for record in dataset.data:
        # Construct the filename based on the specified columns
        filename_parts = [str(record[col]) for col in columns]
        filename = '_'.join(filename_parts) + '.wav'
        file_path = os.path.join(directory_path, filename)

        # Assuming 'audio' key in each record contains the audio segment
        audio_data = record['audio']

        # Save the audio data to a file
        audio_processor.save_audio(file_path, audio_data)


if __name__ == '__main__':
    main()
