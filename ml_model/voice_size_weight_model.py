import json

import keras
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, LSTM
from tensorflow.python.keras import metrics, activations, optimizers
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import MaxPooling2D, Conv2D, Reshape


class VoiceSizeWeightModel:
    def __init__(self, hyperparameters):
        self.model = None
        self.meta_data = {}
        self.last_training_history = None
        self.hyperparameters = hyperparameters

    def create_new_model(self,
                         input_shapes,
                         hyperparameters,
                         size_branch_activation='relu',
                         weight_branch_activation='relu',
                         dense_neurons=64,
                         dense_layers=1,
                         dense_activation='relu',
                         output_activation='linear'):

        self.hyperparameters = hyperparameters

        size_branch_activation = self.fix_activation(size_branch_activation)
        weight_branch_activation = self.fix_activation(weight_branch_activation)

        # Initialize input layers for different feature types
        inputs = []
        branches = []

        # Create size branch
        shape = input_shapes[0]
        layer = Input(shape=shape)
        inputs.append(layer)
        layer = self.create_layer(hyperparameters['model_type_for_size_branch'],
                                  shape,
                                  self.hyperparameters['branch_layers'],
                                  self.hyperparameters['branch_units'],
                                  size_branch_activation,
                                  layer)
        branches.append(layer)

        # # Create weight branch
        # shape = input_shapes[1]
        # layer = Input(shape=shape)
        # inputs.append(layer)
        # layer = self.create_layer(hyperparameters['model_type_for_weight_branch'],
        #                           shape,
        #                           self.hyperparameters['branch_layers'],
        #                           self.hyperparameters['branch_units'],
        #                           weight_branch_activation,
        #                           layer)
        # branches.append(layer)

        # Combine branches if there's more than one
        if len(branches) > 1:
            combined = Concatenate()(branches)
        else:
            combined = branches[0]

        # Final dense layers
        for _ in range(dense_layers):
            combined = Dense(dense_neurons, activation=dense_activation)(combined)

        # Output layer
        output = Dense(1, activation=output_activation)(combined)

        # Create the model
        self.model = Model(inputs=inputs, outputs=output)

        if (hyperparameters['model_type_for_size_branch'] == 'lstm' or
                hyperparameters['model_type_for_weight_branch'] == 'lstm'):
            optimizer = optimizers.adam_v2.Adam(clipvalue=1.0)
            # optimizer = optimizers.adam_v2.Adam(clipnorm=1.0)
        else:
            optimizer = 'adam'

        # Compile the model
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error',
                           metrics=[metrics.MeanAbsoluteError()])

    def create_layer(self, type, shape, layer_count, dense_neurons_or_lstm_units, activation, prev_layer):
        layer = prev_layer
        units = dense_neurons_or_lstm_units

        dimensionality = len(shape)
        if dimensionality > 2:
            raise "Not supported"

        if type == 'dense':
            for _ in range(layer_count):
                layer = Dense(units, activation=activation)(layer)
            layer = Flatten()(layer)
        elif type == 'conv':
            if dimensionality == 1:
                layer = self.create_conv1d_layer(shape, activation, layer)
            else:
                layer = self.create_conv2d_layer(shape, activation, layer)
            layer = Flatten()(layer)
        elif type == 'lstm':
            # TODO: Is it okay for the 2nd dimension in MFCC's etc. to be the feature_dim in an LSTM? If so,
            # is it wrong to just set feature_dim to 1 for 1-dimensional data, like audio?
            if dimensionality == 1:
                layer = self.convert_input_into_sequences(shape, feature_dim=64, prev_layer=layer)
            layer = self.create_lstm_layer(layer_count, units, activation, layer)
        else:
            raise "Unexpected model_type " + type

        return layer

    def create_conv1d_layer(self, shape, activation, prev_layer):
        layer = Reshape(target_shape=(*shape, 1))(prev_layer)  # Add channel dimension
        layer = Conv1D(self.hyperparameters['conv_filters'],
                       kernel_size=self.hyperparameters['kernel_size'][0],
                       activation=activation)(layer)
        layer = MaxPooling1D(pool_size=self.hyperparameters['pool_size'][0])(layer)
        return layer

    def create_conv2d_layer(self, shape, activation, prev_layer):
        layer = Reshape(target_shape=(*shape, 1))(prev_layer)  # Add channel dimension
        layer = Conv2D(self.hyperparameters['conv_filters'], kernel_size=self.hyperparameters['kernel_size'],
                       activation=activation)(layer)
        layer = MaxPooling2D(pool_size=self.hyperparameters['pool_size'])(layer)
        return layer

    def convert_input_into_sequences(self, shape, feature_dim, prev_layer):
        """ feature_dim - Number of features per time step """
        sequence_length = shape[0] // feature_dim  # Number of time steps
        layer = Reshape(target_shape=(sequence_length, feature_dim))(prev_layer)
        return layer

    def create_lstm_layer(self, layers, units, activation, layer):
        for _ in range(layers):
            layer = LSTM(units, activation=activation, return_sequences=True)(layer)
        layer = LSTM(units, activation=activation)(layer)  # Last LSTM layer does not return sequences
        return layer

    def fix_activation(self, activation):
        if activation == 'swish':
            activation = activations.swish
        return activation

    def train(self, input_data, labels, val_data, val_labels, epochs=20, batch_size=32):
        # print("Number of inputs expected by the model:", len(self.model.input))
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001, mode='min', verbose=1)
        history = self.model.fit(input_data,
                                 labels,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(val_data, val_labels),
                                 callbacks=[early_stopping])
        self.last_training_history = history
        return history

    def predict(self, input_data):
        return self.model.predict(input_data)

    def evaluate(self, input_data, labels):
        evaluation = self.model.evaluate(input_data, labels)
        mse, mae = evaluation[0], evaluation[1]  # Assuming first is MSE, second is MAE
        return mse, mae

    def evaluate_predictions(self, input_data, labels):
        predicted = self.predict(input_data)
        actual = labels
        errors = actual - predicted
        return actual, predicted, errors

    def print_summary(self):
        # Print the summary of the model
        self.model.summary()

    def save(self, base_filename):
        # Save the model
        self.model.save(base_filename + '.keras')

        # Save the hyperparameters and input shapes to a JSON file
        with open(base_filename + '.json', 'w') as f:
            json.dump(self.meta_data, f, indent=4)

    @staticmethod
    def load(base_filename):
        # Load the hyperparameters and input shapes from a JSON file
        with open(base_filename + '.json', 'r') as f:
            meta_data = json.load(f)

        # Load the model
        model = keras.models.load_model(base_filename + '.keras')
        voice_model = VoiceSizeWeightModel(meta_data['hyperparameters'])
        voice_model.model = model
        voice_model.meta_data = meta_data
        return voice_model

    def print_sample_predictions(self, features, labels, filenames):
        # Use the model to predict the test dataset
        predictions = self.model.predict(features)

        # Flatten the predictions and actual labels if they are multidimensional
        # predictions_flat = predictions.flatten()
        # labels_flat = labels.flatten()

        # Calculate the absolute error for each sample
        # absolute_errors = np.abs(labels_flat - predictions_flat)

        # Print the MAE for each sample
        print(f"Sample\tMAE\tActual Size\tPredicted Size\tActual Weight\tPredicted Weight\tFilename")
        for i, actual_size, actual_weight, predicted, filename in zip(range(len(labels[0])), labels[0], labels[1],
                                                                      predictions, filenames):
            mae = np.abs(predicted[0] - actual_size) + np.abs(predicted[1] - actual_weight) / 2
            print(f"{i}\t{mae}\t{actual_size}\t{predicted[0]}\t{actual_weight}\t{predicted[1]}\t{filename}")

    def get_layer_activations(self, input_data, layer_index):
        intermediate_model = Model(inputs=self.model.input, outputs=self.model.layers[layer_index].output)
        activations = intermediate_model.predict(input_data)
        return activations

    def plot_loss_over_epochs(self, history):
        plt.ion()  # Turn on interactive mode
        plt.figure()  # Create a new figure
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
        plt.pause(0.001)  # Pause to ensure plot updates

    def plot_actual_vs_predicted(self, actual, predicted):
        plt.ion()
        plt.figure()
        plt.scatter(actual, predicted)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()
        plt.pause(0.001)

    def plot_activations_histogram(self, layer_index, input_data):
        # Create an auxiliary model to get activations from the specified layer
        layer_output = self.model.layers[layer_index].output
        auxiliary_model = Model(inputs=self.model.input, outputs=layer_output)

        # Get activations by predicting with the auxiliary model
        activations = auxiliary_model.predict(input_data)

        plt.ion()
        plt.figure()
        plt.hist(activations, bins='auto')
        plt.title(f'Activations for Layer {layer_index}')
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')
        plt.show()
        plt.pause(0.001)

    def plot_error_distribution(self, errors):
        plt.ion()
        plt.figure()
        plt.hist(errors, bins='auto')
        plt.title('Distribution of Errors')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.show()
        plt.pause(0.001)
