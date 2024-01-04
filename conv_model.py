import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from keras import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


class VoiceModel:
    def __init__(self, feature_length, conv_filters=64, kernel_size=5, pool_size=4, dense_neurons=64):
        # Initialize the model
        self.model = Sequential()

        # Convolutional layer
        self.model.add(
            Conv1D(conv_filters, kernel_size=kernel_size, activation='relu', input_shape=(feature_length, 1)))

        # MaxPooling layer
        self.model.add(MaxPooling1D(pool_size=pool_size))

        # Flatten layer
        self.model.add(Flatten())

        # Dense layer
        self.model.add(Dense(dense_neurons, activation='relu'))

        # Output layer: 2 neurons for size and weight
        self.model.add(Dense(2))

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error',
                           metrics=[metrics.MeanAbsoluteError()])
    def train(self, audio_samples, labels, val_samples, val_labels, epochs=10, batch_size=32):
        history = self.model.fit(audio_samples,
                                 labels,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(val_samples, val_labels))
        return history

    def evaluate(self, audio_samples, labels):
        evaluation = self.model.evaluate(audio_samples, labels)
        mse, mae = evaluation[0], evaluation[1]  # Assuming first is MSE, second is MAE
        return mse, mae

    def predict(self, audio_sample):
        # Predict using the model
        return self.model.predict(audio_sample)

    def print_summary(self):
        # Print the summary of the model
        self.model.summary()

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

    def save(self, filename):
        # Save the model
        self.model.save(filename)

    def load(self, filename):
        # Load the model
        self.model = keras.models.load_model(filename)

# Example of using the class
# feature_length = 100  # Replace with the actual feature length
# voice_model = VoiceModel(feature_length=100, conv_filters=32, kernel_size=3, pool_size=2, dense_neurons=128)

# Example of training
# voice_model.train(training_data, training_labels)

# Example of evaluating
# evaluation_result = voice_model.evaluate(test_data, test_labels)

# Example of predicting
# prediction = voice_model.predict(sample_data)

# Example of saving
# voice_model.save('voice_model.h5')

# Example of loading
# voice_model.load('voice_model.h5')
