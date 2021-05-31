from unittest import TestCase
from vader import Vader, analyze_polarity_scores
import pandas as pd
from main_app import *


class TestMain(TestCase):
	def test_lowercase_text(self):
		expected = lowercase_text("This is a test")
		self.assertEqual(expected, "this is a test")

	def test_remove_punctuation(self): # Punctuation should be removed.
		expected = remove_punctuation("Question? Exclamation!")
		self.assertEqual(expected, "Question Exclamation")

	def test_remove_links_and_emails(self): # This test doesn't work and it needs to be fixed.
		expected = remove_links_and_emails("abcdef https://stackoverflow.com/ fdsffdf")
		self.assertEqual(expected, "abcdef fdsffdf")


	def test_correct_spelling(self):
		d = enchant.Dict("en_US")
		expected = correct_spelling(d, "thsi is not speled correctli")
		self.assertEqual(expected, 3)
