import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

model = keras.Sequential(
    [
        # 1st Convolutional Layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # 2nd Convolutional Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # 3rd Convolutional Layer
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Flattening
        layers.Flatten(),

        # Fully Connected Layers
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        # Output Layer (binary classification)
        layers.Dense(1, activation='sigmoid'),
    ]
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values between 0 and 1
    rotation_range=20,         # Rotate images by up to 20 degrees
    width_shift_range=0.2,     # Shift width by up to 20%
    height_shift_range=0.2,    # Shift height by up to 20%
    zoom_range=0.2,            # Zoom in by up to 20%
    horizontal_flip=True,      # Flip images horizontally
    fill_mode='nearest'        # Fill missing pixels after transformation
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Directory containing images
train_generator = train_datagen.flow_from_directory(
    'C:/Users/tomge/Desktop/Python code/datasets/wildfire-satellite/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)


validation_generator = test_datagen.flow_from_directory(
    'C:/Users/tomge/Desktop/Python code/datasets/wildfire-satellite/valid',            # Path to the test/validation data
    target_size=(128, 128),    # Resize images to 128x128
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'C:/Users/tomge/Desktop/Python code/datasets/wildfire-satellite/test',            # Path to the test/validation data
    target_size=(128, 128),    # Resize images to 128x128
    batch_size=32,
    class_mode='binary'
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

history = model.fit(
    train_generator,
    epochs=1,                           # Number of training epochs
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint],
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)

print(f'Test accuracy: {test_acc * 100:.2f}%')
model.save('forest_fire_model.keras')


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
