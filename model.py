import keras

def create_classifier():
    model = keras.Sequential([
        # 1st Convolutional Layer
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
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
