from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import load_model
from PIL import ImageFile
from model import create_classifier
import datasets

if __name__ == '__main__':
    while True:
        try:
            f = open("forest_fire_model.keras")
        except FileNotFoundError:
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            model = create_classifier()
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

            history = model.fit(
                datasets.train_ds,
                epochs=10,                           # Number of training epochs
                validation_data=datasets.val_ds,
                callbacks=[early_stopping, model_checkpoint],
            )

            # Evaluate the model on the test set
            test_loss, test_acc = model.evaluate(datasets.test_ds)

            print(f'Test accuracy: {test_acc * 100:.2f}%')
            model.save('forest_fire_model.keras')
        else:
            model = load_model('forest_fire_model.keras')

            f.close()
            break
