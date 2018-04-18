import numpy as np
from keras.preprocessing.image import img_to_array


def preprocess(img, target_size):
    """Preprocess the image for model prediction."""
    # Greyscale and resize
    img = img.convert('L').resize(target_size)
    # Feature scale
    img = (img_to_array(img) / 255. - .5) * 2.
    # Add batch dim i.e. model expects arrays of shape (batch, height, width, channels)
    return np.expand_dims(img, 0)
