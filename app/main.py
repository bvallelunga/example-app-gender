import os
import binascii

import keras.models

from .utils import base64_to_img
from .utils import preprocess

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'gender_mini_XCEPTION.21-0.95.hdf5')
LABELS = ('woman', 'man')
SCORE_PRECISION = 2


class ModelInterface(object):
    def __init__(self, model=None):
        if model is None:
            model = keras.models.load_model(MODEL_PATH)
        self.model = model

    def predict(self, input):
        if 'face' not in input:
            raise KeyError("No key named 'face' in input.")
        try:
            face = base64_to_img(input['face'])
        except (binascii.Error, AttributeError):
            raise ValueError("'face' must be a base64 encoded string.")
        min_dim = self.model.input_shape[1]
        if any(dim < min_dim for dim in face.size):
            raise ValueError("'face' can not have a height or width less than {} pixels.".format(min_dim))

        face = preprocess(face, self.model.input_shape[1:3])
        scores = self.model.predict(face).tolist()[0]
        return {'scores': {label: round(score, SCORE_PRECISION) for label, score in zip(LABELS, scores)}}
