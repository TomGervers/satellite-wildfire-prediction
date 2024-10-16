import tensorflow as tf
import numpy as np
from struct import unpack
import os

# Function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size=(128, 128)):
    """
    Load and preprocess an image from the given path.

    This function loads an image, resizes it to the specified target size, 
    normalizes the pixel values to the range [0, 1], and expands its dimensions 
    to create a batch of size 1.

    Args:
        img_path (str): The file path to the image to be loaded.
        target_size (tuple, optional): The target size to resize the image. 
                                        Defaults to (128, 128).

    Returns:
        np.ndarray: A numpy array representation of the preprocessed image 
                    with shape (1, height, width, channels) or None if an error occurs.
    """
    try:
        # Load the image using TensorFlow utility
        img = tf.keras.utils.load_img(img_path, target_size=target_size)
    
        # Convert the image to a numpy array
        img_array = tf.keras.utils.img_to_array(img)
    
        # Normalize the image (same as during training)
        img_array = img_array / 255.0
    
        # Expand dimensions to create a batch (batch size 1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    
    except:
        return None


class JPEG:
    """
    A class to represent a JPEG image file.

    Attributes:
        img_data (bytes): Raw byte data of the JPEG image.

    Methods:
        decode(): Decodes the JPEG image data to verify its integrity.
    """
    def __init__(self, image_file):
        """
        Initialize the JPEG object by reading the image file.

        Args:
            image_file (str): The path to the JPEG image file to be read.
        """
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
        
    def decode(self):
        """
        Decode the JPEG image data.

        This method processes the JPEG markers to ensure the image file is valid. 
        It raises a TypeError if there are issues reading the JPEG file.

        Returns:
            None
        """
        data = self.img_data
        while True:
            marker, = unpack(">H", data[0:2])
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2 + lenchunk:]            
            if len(data) == 0:
                raise TypeError("issue reading jpeg file")  
                
def remove_corrupt(path):
    """
    Remove corrupt image files from the specified directory.

    This function checks all images in the given directory (including subdirectories)
    for corruption using the JPEG class. If a corrupt image is found, it is deleted.

    Args:
        path (str): The directory path containing subdirectories of images to be checked.

    Returns:
        None
    """
    # List all subdirectories in directory
    list_subfolders_with_paths = [f.name for f in os.scandir(path) if f.is_dir()]

    for dir in list_subfolders_with_paths:
        directory = os.path.join(path, dir)
        image_paths = os.listdir(directory)

        for img_path in image_paths:
            full_image_path = os.path.join(directory, img_path)
            image = JPEG(full_image_path) 
            try:
                image.decode()   
            except:
                os.remove(full_image_path) #If image could not be decoded as JPEG, remove it
