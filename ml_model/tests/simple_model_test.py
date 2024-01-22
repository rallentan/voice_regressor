from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Flatten, Dense
from tensorflow.python.keras import metrics
from tensorflow.python.keras.layers import Conv2D, Reshape


def create_test_model(input_shape, use_reshape=False, conv_filters=16, kernel_size=(3, 3)):
    # Define input layer
    input_layer = Input(shape=input_shape)

    # Optional Reshape layer
    if use_reshape:
        x = Reshape(target_shape=(*input_shape, 1))(input_layer)
    else:
        x = input_layer

    # Conv2D layer
    x = Conv2D(conv_filters, kernel_size=kernel_size, activation='relu')(x)

    # Flatten and output layer
    x = Flatten()(x)
    output_layer = Dense(1, activation='linear')(x)

    # Create and compile the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.MeanAbsoluteError()])

    return model


# Example usage
input_shape = (20, 20, 1)  # Example input shape for Conv2D without Reshape
model = create_test_model(input_shape, use_reshape=False)

# Test saving the model
model.save('test_model.keras')
print("Model saved successfully.")