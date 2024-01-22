import json
import os
import sys

import numpy as np

from mci_lib_ml.basic_operations import Ops
from voice_size_weight_model import VoiceSizeWeightModel
from mci_lib_ml.hyperparameter_optimizer import HyperparameterOptimizer
from data_preparation.audio_processor import AudioPreprocessor
from scipy.optimize import curve_fit

from ml_model.audio_dataset import AudioDataSet
from ml_model.data_preprocessor import DataPreprocessor

RAW_AUDIO_DIRECTORY = r'C:\Users\User\Documents\Sound recordings'
SEGMENTED_AUDIO_DIRECTORY = r'C:\Users\User\Documents\Sound recordings\segmented'
PROCESSED_AUDIO_DIRECTORY = r'C:\Users\User\Documents\Sound recordings\processed'
DATASET_CACHE_BASE_FILENAME = 'voice_dataset'
AUDIO_SAVE_FILE_LABELS = ['actor', 'size', 'weight', 'sound', 'sound detail', 'sound source', 'index', 'augmentations']

audio_processor = AudioPreprocessor(sample_rate=44100)

# TODO: Consider using pooling for the LSTM layer
# TODO: Make the choice of whether to use MaxPooling2D a hyperparameter
# TODO: Add regularization to the model to reduce sensitivity to initial weights
# TODO: Add a hyperparameter for random_seed to randomize yet retain the initial weights of the model
# TODO: Prune bad samples


def main(mode):
    if mode == 'generate-dataset-cache':
        generate_dataset_cache()
    elif mode == 'optimize-hyperparameters':
        optimize_hyperparameters()
    elif mode == 'view-hyperparameter-charts':
        view_hyperparameter_charts()
    elif mode == 'train-best-model':
        train_best_model()
    elif mode == 'view-best-model':
        view_best_model()


def generate_dataset_cache():
    model_config = get_last_best_model_config()
    hyperparameters = config_to_hyperparameters(model_config)

    # Generate and process all possible input data to the model for the cache,
    # so it can be used for various permutations of hyperparameters.
    hyperparameters['use_raw_audio'] = True
    hyperparameters['use_mfcc'] = True
    hyperparameters['use_spectrogram'] = True
    hyperparameters['use_hnr_mean_and_std'] = True
    hyperparameters['use_hnr_over_time'] = True
    hyperparameters['use_spectral'] = True
    hyperparameters['use_wavelet_at_level'] = True
    hyperparameters['use_wavelets_mean_and_std'] = True

    for normalize_volume in [True, False]:
        hyperparameters['normalize_volume_and_amplitude'] = normalize_volume
        preprocessor = DataPreprocessor(hyperparameters)

        dataset = preprocessor.load_dataset_from_directory()
        dataset = preprocessor.preprocess_for_training(dataset)

        print('Label encodings: ', dataset.encoders)

        dataset_filename = get_cache_filename(hyperparameters)
        dataset.save_to_file(dataset_filename)
        print('Dataset cache saved to file: ' + dataset_filename)


def optimize_hyperparameters():
    optimizer = HyperparameterOptimizer(objective_function, "study.pickle")

    if optimizer.try_load_study():
        print("Existing study found; loaded")
    else:
        print("Starting new study")
        optimizer.create_new_study()

    optimizer.optimize(n_trials=80)


def view_hyperparameter_charts():
    optimizer = HyperparameterOptimizer(objective_function, "study.pickle")

    if not optimizer.try_load_study():
        raise "No study found"

    optimizer.print_all_trials()
    optimizer.print_trial_param_counts()
    optimizer.print_best_trial()
    optimizer.show_visualizations()
    optimizer.show_pca_graphs(n_components=2)


def train_best_model():
    model_config = get_last_best_model_config()
    hyperparameters = config_to_hyperparameters(model_config)

    voice_model, test_features, test_labels = train_model(hyperparameters)
    voice_model.save(base_filename='trained_voice_model')
    print('Model saved.')


def view_best_model():
    # Look for samples that are unpredictable indicating they are bad training data
    voice_model = VoiceSizeWeightModel.load('trained_voice_model')
    hyperparameters = voice_model.meta_data['hyperparameters']
    dataset = AudioDataSet.load_from_file(get_cache_filename(hyperparameters))
    preprocessor = DataPreprocessor(hyperparameters)
    feature_columns = preprocessor.get_feature_columns()
    filenames = [preprocessor.sample_to_filename(sample, AUDIO_SAVE_FILE_LABELS, '.wav') for sample in dataset.data]
    features = dataset.get_columns(feature_columns)
    labels = dataset.get_columns(['size', 'weight'])
    voice_model.print_sample_predictions(features.to_numpy_arrays(), np.array(labels.to_numpy_arrays()), filenames)

    # # Get metrics to display
    # actual, predicted, errors = model.evaluate_predictions(test_features, test_labels)
    # predictions = model.model.predict(test_features)
    #
    # # Plot graphs
    # model.plot_actual_vs_predicted(test_labels, predictions)
    # model.plot_loss_over_epochs(history)
    # model.plot_actual_vs_predicted(actual, predicted)
    # #model.plot_activations_histogram(1, test_features)
    # #model.plot_error_distribution(errors)


# noinspection PyDictCreation
def objective_function(trial):
    # Define the hyperparameter search space using the trial object
    model_config = get_last_best_model_config()
    model_config = model_config | {
        'normalize_volume_and_amplitude': trial.suggest_categorical('normalize_volume_and_amplitude', [True, False]),
        'model_type_for_size_branch': trial.suggest_categorical('model_type_for_size_branch',
                                                                ['dense', 'conv', 'lstm']),
        'model_type_for_weight_branch': trial.suggest_categorical('model_type_for_weight_branch',
                                                                  ['dense', 'conv', 'lstm']),
        'inputs_for_size_branch': trial.suggest_categorical('inputs_for_size_branch',
                                                            ['spectrogram', 'mfcc']),
        'inputs_for_weight_branch': trial.suggest_categorical('inputs_for_weight_branch',
                                                              ['hnr', 'wavelet', 'spectral', 'raw_audio']),
        # 'conv_filters': trial.suggest_categorical('conv_filters', [8, 16, 32, 64]),
        # 'kernel_size': trial.suggest_categorical('kernel_size', [(3, 3), (5, 5)]),
        # 'pool_size': trial.suggest_categorical('pool_size', [(2, 2), (3, 3), (5, 5), (8, 8)]),
        'wavelet_level': trial.suggest_categorical('wavelet_level', [5]),
        'branch_layers': trial.suggest_categorical('branch_layers', [1, 2, 3]),
        'branch_units': trial.suggest_categorical('branch_units', [4, 16, 64]),
        'size_branch_activation_function': trial.suggest_categorical('size_branch_activation_function',
                                                                     ['relu', 'leaky_relu', 'elu', 'swish', 'tanh']),
        'weight_branch_activation_function': trial.suggest_categorical('weight_branch_activation_function',
                                                                       ['relu', 'leaky_relu', 'elu', 'swish', 'tanh']),
        'dense_layers': trial.suggest_categorical('dense_layers', [1, 2, 3]),
        'dense_neurons': trial.suggest_categorical('dense_neurons', [4, 8, 16, 64, 128]),
        'dense_activation_function': trial.suggest_categorical('dense_activation_function',
                                                               ['relu', 'leaky_relu', 'elu', 'swish', 'tanh']),
        'output_activation': trial.suggest_categorical('output_activation', ['relu', 'linear'])}

    hyperparameters = config_to_hyperparameters(model_config)

    # Train the model with these hyperparameters
    model, test_features, test_labels = train_model(hyperparameters)

    # Return the metric to be minimized
    return model.meta_data['val_loss']


def get_last_best_model_config():
    return {
        'normalize_volume_and_amplitude': True,
        'mfcc_coefficients': 13,
        'model_type_for_size_branch': 'conv',
        'model_type_for_weight_branch': 'lstm',
        'inputs_for_size_branch': 'spectrogram',
        'inputs_for_weight_branch': 'hnr',
        'conv_filters': 16,
        'kernel_size': (3, 3),
        'pool_size': (2, 2),
        'wavelet_level': 5,
        'branch_layers': 1,
        'branch_units': 64,
        'dense_layers': 3,
        'dense_neurons': 16,
        'size_branch_activation_function': 'tanh',
        'weight_branch_activation_function': 'tanh',
        'dense_activation_function': 'swish',
        'output_activation': 'relu',
        'epochs': 40,
        'batch_size': 32,
    }


def config_to_hyperparameters(model_config):
    hyperparameters = dict.copy(model_config)

    # Propagate the optimizers choice of model_type into the preprocessor parameters
    size_inputs = hyperparameters['inputs_for_size_branch']
    weight_inputs = hyperparameters['inputs_for_weight_branch']
    hyperparameters['use_raw_audio'] = (size_inputs == 'raw_audio' or weight_inputs == 'raw_audio')
    hyperparameters['use_mfcc'] = (size_inputs == 'mfcc' or weight_inputs == 'mfcc')
    hyperparameters['use_spectrogram'] = (size_inputs == 'spectrogram' or weight_inputs == 'spectrogram')
    hyperparameters['use_hnr_mean_and_std'] = (size_inputs == 'hnr_mean_and_std' or weight_inputs == 'hnr_mean_and_std')
    hyperparameters['use_hnr_over_time'] = (size_inputs == 'hnr_over_time' or weight_inputs == 'hnr_over_time')
    hyperparameters['use_spectral'] = (size_inputs == 'spectral' or weight_inputs == 'spectral')
    hyperparameters['use_wavelet_at_level'] = (size_inputs == 'wavelet_at_level' or weight_inputs == 'wavelet_at_level')
    hyperparameters['use_wavelets_mean_and_std'] = (size_inputs == 'wavelets_mean_and_std' or
                                                    weight_inputs == 'wavelets_mean_and_std')

    return hyperparameters


def train_model(hyperparameters):
    print("Hyperparameters:\n", json.dumps(hyperparameters, indent=4))

    preprocessor = DataPreprocessor(hyperparameters)
    feature_columns = preprocessor.get_feature_columns()

    dataset_filename = get_cache_filename(hyperparameters)

    if os.path.isfile(dataset_filename):
        print("Pre-generated dataset cache found; loading...")
        dataset = AudioDataSet.load_from_file(dataset_filename)
    else:
        print("Preprocessing data...")
        dataset = preprocessor.load_dataset_from_directory()
        dataset = preprocessor.preprocess_for_training(dataset)

    # Split the dataset into training, validation, and test sets
    train, val, test = dataset.split(train_size=0.7,
                                     val_size=0.15,
                                     test_size=0.15,
                                     stratify_cols=['size', 'weight'])

    # Get features and labels for training
    train_features = train.get_columns(feature_columns)
    train_labels = train.get_columns(['size'])
    val_features = val.get_columns(feature_columns)
    val_labels = val.get_columns(['size'])
    test_features = test.get_columns(feature_columns)
    test_labels = test.get_columns(['size'])

    input_shapes = train_features.get_all_column_shapes()
    # Print shape of each input feature
    for feature, shape in input_shapes.items():
        print(f"Shape for '{feature}' feature: {shape}")
    input_shapes = Ops.dict_to_lists(input_shapes)

    # Check label shapes
    print("Shape of labels: ", train_labels.get_shape())

    # Initialize the voice model (assuming feature_length is defined correctly)
    voice_model = VoiceSizeWeightModel(hyperparameters)
    voice_model.create_new_model(input_shapes,
                                 hyperparameters,
                                 size_branch_activation=hyperparameters['size_branch_activation_function'],
                                 weight_branch_activation=hyperparameters['weight_branch_activation_function'],
                                 dense_neurons=hyperparameters['dense_neurons'],
                                 dense_layers=hyperparameters['dense_layers'],
                                 dense_activation=hyperparameters['dense_activation_function'],
                                 output_activation=hyperparameters['output_activation'])

    train_features = train_features.to_numpy_arrays()
    train_labels = train_labels.to_numpy_arrays()
    val_features = val_features.to_numpy_arrays()
    val_labels = val_labels.to_numpy_arrays()
    test_features = test_features.to_numpy_arrays()
    test_labels = test_labels.to_numpy_arrays()

    # Train the model
    voice_model.train(train_features[0],
                      train_labels,
                      val_features[0],
                      val_labels,
                      epochs=hyperparameters['epochs'],
                      batch_size=hyperparameters['batch_size'])

    # Evaluate the model
    mse, mae = voice_model.evaluate(val_features[0], val_labels)

    # Save the model metadata
    voice_model.meta_data = {
        'hyperparameters': hyperparameters,
        'input_shapes': input_shapes,
        'label_encodings': dataset.encoders,
        'val_loss': mse,
        'val_mse': mse,
        'val_mae': mae,
    }

    return voice_model, test_features, test_labels


def get_cache_filename(hyperparameters):
    if hyperparameters['normalize_volume_and_amplitude']:
        dataset_filename = DATASET_CACHE_BASE_FILENAME + '_norm-vol' + '.cache'
    else:
        dataset_filename = DATASET_CACHE_BASE_FILENAME + '_no-norm-vol' + '.cache'
    return dataset_filename


def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def logarithmic_decay(x, a, b, c):
    return a * np.log(b * x + 1) + c


def calculate_learning_curve_consistency_exponential(val_losses):
    # Prepare data for curve fitting (epoch numbers as X, validation losses as y)
    X = np.array(range(len(val_losses)))
    y = np.array(val_losses)

    # Fit the exponential decay function to the data
    try:
        params = curve_fit(exponential_decay, X, y, maxfev=10000)[0]
    except RuntimeError:
        # In case the curve fitting does not converge, return a default low score
        return 0

    # Predict values using the fitted function
    y_pred = exponential_decay(X, *params)

    # Calculate the R² score
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return r2


if __name__ == '__main__':
    arg_mode = None
    if len(sys.argv) > 1:
        arg_mode = sys.argv[1]
    main(arg_mode)
