import base64
from io import BytesIO

import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array


def preprocess(img, target_size):
    """Preprocess the image for model prediction."""
    # Greyscale and resize
    img = img.convert('L').resize(target_size)
    # Feature scale
    img = (img_to_array(img) / 255. - .5) * 2.
    # Add batch dim i.e. model expects arrays of shape (batch, height, width, channels)
    return np.expand_dims(img, 0)


def base64_to_img(string):
    """Convert a base64 encoded string to an image."""
    img_buffer = BytesIO(base64.b64decode(string.encode()))
    return Image.open(img_buffer)
