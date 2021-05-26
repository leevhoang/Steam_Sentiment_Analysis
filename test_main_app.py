from unittest import TestCase
from vader import Vader, analyze_polarity_scores
import pandas as pd
from main_app import *


class TestMain(TestCase):
	def test_lowercase_text(self):
		expected = lowercase_text("This is a test")
		self.assertEqual(expected, "this is a test")

	def test_remove_punctuation(self):
		expected = remove_punctuation("Question? Exclamation!")
		self.assertEqual(expected, "Question Exclamation")

	def test_remove_links_and_emails(self):
		expected = remove_links_and_emails("abcdef https://stackoverflow.com/ fdsffdf")
		self.assertEqual(expected, "abcdef fdsffdf")
