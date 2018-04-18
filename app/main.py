import keras.models
import os
from keras.preprocessing.image import img_to_array
import numpy as np


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'gender_mini_XCEPTION.21-0.95.hdf5')
LABELS = ('woman', 'man')


class ModelInterface(object):
    def __init__(self):
        self.model = keras.models.load_model(MODEL_PATH)

    def predict(self, img):
        img = img.convert('L').resize((64, 64))
        img = img_to_array(img) / 255.
        img = (img - .5) * 2.
        scores = self.model.predict(np.expand_dims(img, 0)).tolist()[0]
        return {'scores': {label: score for label, score in zip(LABELS, scores)}}
