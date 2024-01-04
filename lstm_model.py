import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Bidirectional

# Define the model architecture
model = Sequential()

# Assuming 'timesteps' is the number of time steps in your feature data,
# and 'feature_length' is the number of features at each time step.
input_shape = (timesteps, feature_length)

# Bidirectional LSTM layers
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))

# Dense layers for final output
model.add(Dense(64, activation='relu'))
model.add(Dense(2))  # Output layer: 2 neurons for size (resonance) and weight (constriction)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Model summary
model.summary()
