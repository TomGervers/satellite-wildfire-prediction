import keras

def create_classifier():
    """
    Create a Convolutional Neural Network (CNN) classifier model.

    This function constructs a sequential model for binary classification tasks.
    It includes convolutional layers for feature extraction, max pooling layers for 
    downsampling, and fully connected layers for final predictions.

    Returns:
        keras.Model: A compiled Keras Sequential model ready for training.
    """
    model = keras.Sequential([
        # Input layer defining the shape of the input
        keras.layers.Input(shape=(128, 128, 3)),

        # 1st Convolutional Layer
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # 2nd Convolutional Layer
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # 3rd Convolutional Layer
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flattening
        keras.layers.Flatten(),
        
        # Fully Connected Layers
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        
        # Output Layer (binary classification)
        keras.layers.Dense(1, activation='sigmoid'),  # Output layer
    ])
    
    return model
