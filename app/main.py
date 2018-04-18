import os

import keras.models

from .utils import base64_to_img
from .utils import preprocess

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'gender_mini_XCEPTION.21-0.95.hdf5')
LABELS = ('woman', 'man')


class ModelInterface(object):
    def __init__(self, model=None):
        if model is None:
            model = keras.models.load_model(MODEL_PATH)
        self.model = model

    def predict(self, input):
        face = base64_to_img(input['face'])
        face = preprocess(face, self.model.input_shape[1:3])
        scores = self.model.predict(face).tolist()[0]
        return {'scores': {label: score for label, score in zip(LABELS, scores)}}
