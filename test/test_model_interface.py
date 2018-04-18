import os
import unittest

import keras.models

from app.main import MODEL_PATH
from app.main import ModelInterface
from app.main import SCORE_PRECISION
from .utils import img_to_base64

MODEL = keras.models.load_model(MODEL_PATH)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG = img_to_base64(os.path.join(BASE_PATH, 'test.jpg'))


class TestModelInterface(unittest.TestCase):
    def setUp(self):
        self.interface = ModelInterface(MODEL)

    def tearDown(self):
        self.interface = None

    def test_output_has_scores_key(self):
        """The output dict must have a 'scores' key."""
        result = self.interface.predict({'face': IMG})
        self.assertIn('scores', result)

    def test_scores_has_woman_key(self):
        """The scores dict must have a 'woman' key."""
        scores = self.interface.predict({'face': IMG})['scores']
        self.assertIn('woman', scores)

    def test_scores_has_man_key(self):
        """The scores dict must have a 'man' key."""
        scores = self.interface.predict({'face': IMG})['scores']
        self.assertIn('man', scores)

    def test_scores_are_floats(self):
        """The score values must be floats."""
        scores = self.interface.predict({'face': IMG})['scores']
        self.assertTrue(all(isinstance(score, float) for score in scores.values()))

    def test_scores_have_correct_precision(self):
        """The scores must have the correct precision."""
        scores = self.interface.predict({'face': IMG})['scores']
        self.assertTrue(all(score == round(score, SCORE_PRECISION) for score in scores.values()))
