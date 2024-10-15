import keras

# Set the image size and batch size
image_size = (128, 128)
batch_size = 32

# Load the datasets
train_ds = keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

val_ds = keras.utils.image_dataset_from_directory(
    'data/valid',
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    'data/test',
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False
)

normalization_layer = keras.layers.Rescaling(1./255)

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.2),
])

# Apply normalization and augmentation to the training dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x)), y))

# Apply normalization only to the validation and test datasets
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
