from unittest import TestCase
from vader import Vader, analyze_polarity_scores
import pandas as pd

class TestVader(TestCase):
    def test_vader_analysis_true(self):
        vader = Vader()
        reviews = ["story is great but graphic looks like mafia 2 classic"]
        self.assertEqual(vader.vader_analysis(reviews), [True])

    def test_vader_analysis_false(self):
        vader = Vader()
        reviews = ["story is so bad"]
        self.assertEqual(vader.vader_analysis(reviews), [False])

    def test_analyze_polarity_scores(self):
        reviews = "story is great but graphic looks like mafia 2 classic"
        expected = {'compound': 0.7003, 'neg': 0.0, 'neu': 0.547, 'pos': 0.453}
        self.assertEqual(analyze_polarity_scores(reviews), expected)

    def test_vader_validation_100(self):
        vader = Vader()
        y_train = pd.Series([True, False, True, False])
        vader.analysis_results = [True, False, True, False]
        self.assertEqual((vader.vader_validation(y_train)), 1.0)

    def test_vader_validation_0(self):
        vader = Vader()
        y_train = pd.Series([True, False, True, False])
        vader.analysis_results = [False, True, False, True]
        self.assertEqual((vader.vader_validation(y_train)), 0.0)
