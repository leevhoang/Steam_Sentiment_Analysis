from unittest import TestCase
from vader import Vader, analyze_polarity_scores
import pandas as pd
from main_app import lowercase_text


class TestMain(TestCase):
    def test_lowercase_text(self):
        expected = lowercase_text("This is a test")
        self.assertEqual(expected, "this is a test")
