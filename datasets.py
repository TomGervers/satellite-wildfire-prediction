import keras
#import PIL #Might be needed depending on your data
from utils import remove_corrupt

#PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True #Might be needed depending on your data

# Set the image size and batch size
image_size = (128, 128)
batch_size = 32  

# Normalization layer to scale pixel values to be between 0 and 1
normalization_layer = keras.layers.Rescaling(1./255)

# Data augmentation layer to apply random transformations to images
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.2),
])

def create_train_dataset():
    """
    Create and return the training dataset after removing corrupt images.

    This function loads images from the 'data/train' directory, applies
    data augmentation and normalization, and returns a batched dataset.

    Returns:
        tf.data.Dataset: A dataset object containing the augmented and normalized training images.
    """
    #Remove corrupt images
    remove_corrupt('data/train')

    # Load the datasets
    train_ds = keras.utils.image_dataset_from_directory(
        'data/train',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

    # Apply normalization and augmentation to the training dataset
    train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x)), y))

    return train_ds

def create_val_dataset():
    """
    Create and return the validation dataset after removing corrupt images.

    This function loads images from the 'data/valid' directory and applies
    normalization. It returns a batched dataset without data augmentation.

    Returns:
        tf.data.Dataset: A dataset object containing the normalized validation images.
    """
    remove_corrupt('data/valid')

    val_ds = keras.utils.image_dataset_from_directory(
        'data/valid',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Apply normalization only to the validation and test datasets
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return val_ds

def create_test_dataset():
    """
    Create and return the test dataset after removing corrupt images.

    This function loads images from the 'data/test' directory and applies
    normalization. It returns a batched dataset without data augmentation.

    Returns:
        tf.data.Dataset: A dataset object containing the normalized test images.
    """
    remove_corrupt('data/test')

    test_ds = keras.utils.image_dataset_from_directory(
        'data/test',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Apply normalization only to the validation and test datasets
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return test_ds