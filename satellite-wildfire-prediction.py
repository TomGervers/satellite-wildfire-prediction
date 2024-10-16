import tensorflow as tf
import keras
#import PIL #Might be needed depending on your data
import matplotlib.pyplot as plt
import logging
import os
from model import create_classifier
import datasets
import utils

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_PATH = 'forest_fire_model.keras'
BEST_MODEL_PATH = 'best_model.keras'
PREDICTIONS_FILE = 'predictions.txt'
IMAGE_DIRECTORY = 'sample_images'
EPOCHS = 10

def load_or_train_model():
    """
    Load an existing model from a file or train a new model if it doesn't exist.

    Returns:
        keras.Model: The loaded or newly trained Keras model.
    """
    # If the model file exists
    if os.path.exists(MODEL_PATH):
        # Load the model file
        logging.info(f'Loading model from {MODEL_PATH}')
        return keras.saving.load_model(MODEL_PATH)
    # If the model file does not exist
    else:
        logging.info('Training a new model...')

        # Load model structure and compile the model
        model = create_classifier()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Create early stopping and checkpoint callbacks
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model_checkpoint = keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True)

        # Create the datasets
        train_ds = datasets.create_train_dataset()
        val_ds = datasets.create_val_dataset()

        # Fit the model
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            callbacks=[early_stopping, model_checkpoint],
        )

        # Evaluate the model on the test set
        test_ds = datasets.create_test_dataset()
        test_loss, test_acc = model.evaluate(test_ds)
        logging.info(f'Test accuracy: {test_acc * 100:.2f}%')

        # Save the model and return it
        model.save(MODEL_PATH)
        return model

def predict_images(model):
    """
    Make predictions on images located in the specified directory.

    Args:
        model (keras.Model): The trained Keras model to use for making predictions.

    Returns:
        None: The function writes predictions to a text file and optionally displays the images.
    """
    image_files = [f for f in os.listdir(IMAGE_DIRECTORY) if f.endswith(('png', 'jpg', 'jpeg'))]

    with open(PREDICTIONS_FILE, 'w') as f:
        for image_file in image_files:
            img_path = os.path.join(IMAGE_DIRECTORY, image_file)
            preprocessed_image = utils.load_and_preprocess_image(img_path)
            prediction = model.predict(preprocessed_image)

            result = "Prediction: Wildfire present" if prediction[0] > 0.5 else "Prediction: No wildfire present"
            output = f"Image: {image_file} - {result}"
            logging.info(output)
            f.write(output + '\n')

            # Uncomment to display the image with the corresponding prediction
            """
            img = tf.keras.utils.load_img(img_path)  # Load image for display
            plt.imshow(img)
            plt.title(result)
            plt.axis('off')
            plt.show()
            """

def main():
    """Main function to load or train the model and make predictions on images."""
    model = load_or_train_model()
    predict_images(model)

if __name__ == '__main__':
    main()
