# 4/25/2021
# CS 510 - Adventures in NLP
# Steam Sentiment Analysis
#
# This file reads and preprocesses reviews from the Steam Game Reviews dataset.
# The dataset can be found here: https://www.kaggle.com/smeeeow/steam-game-reviews
# To use this file, download the dataset from the above link and extract its contents to the same directory as main_app.py.
#
# Source for Embedding layer and preprocessing info: https://realpython.com/python-keras-text-classification/#introducing-keras
# Source for using Embedding layer with LSTM: https://www.liip.ch/en/blog/sentiment-detection-with-keras-word-embeddings-and-lstm-deep-learning-networks

# Import statements
import pandas as pd  # For reading CSV files
import matplotlib.pyplot as plt  # For making plots
import os
import time  # For timing how long it takes to run the script.
import numpy as np

# SKlearn - For splitting the dataset into the training and testing sets.
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Import neural network code from a separate python file
# 5/10/2021
from steam_nn import define_model, train_model

from tensorflow.keras.preprocessing.text import Tokenizer  # For preprocessing text
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Vader import
from vader import Vader
from bert_analysis import TEXT_MODEL, training_model

import enchant
import string

# UNUSED IMPORTS

# Spell checker
# from autocorrect import Speller #correcting the spellings 
# from langdetect import detect

# # Enforce consistent results from langdetect
# from langdetect import DetectorFactory
# DetectorFactory.seed = 0

# from textblob import TextBlob



# Path to all CSVs
DATA_PATH = 'game_rvw_csvs'


# ====================================================================================================
# LOAD DATA
# ====================================================================================================


# Read all reviews in DATA_PATH
def read_all_reviews():
	# Empty dataframe to hold all reviews.
	all_reviews = []
	length = 0

	# Array containing a list of CSV files:
	review_data = os.listdir(DATA_PATH)

	# # DEBUG - Read only a subset of reviews.
	# review_data = review_data[0:1] 

	# This is a list of review csvs that were completely empty.
	empty_reviews = ['50100_SidMeiersCivilizationV.csv', '255710_CitiesSkylines.csv', '292030_TheWitcher3WildHunt.csv',
	                 '435150_DivinityOriginalSin2.csv', '1145360_Hades.csv', '1222730_STARWARSSquadrons.csv']
	review_data = [filename for filename in review_data if filename not in empty_reviews]

	# For all reviews in the data path, read and load them into a single dataframe.
	for reviews in review_data:
		print("Reading " + reviews)
		data = pd.read_csv(DATA_PATH + "/" + reviews)
		length += data.shape[0]
		all_reviews.append(data)

	# Create the dataframe and return it.
	all_reviews = pd.concat(all_reviews, axis=0, ignore_index=True)
	print("Total number of rows: {}".format(length))
	return all_reviews


# ====================================================================================================
# PREPROCESSING
# ====================================================================================================

# Lowercase a single review
def lowercase_text(review):
	return review.lower()


# Remove all punctuation from a single review (. , ? ! ;:)
# Replace all punctuation with whitespace
def remove_punctuation(review):
	punctuation = """.,?!;:"""

	for char in review:
		if char in punctuation:
			review = review.replace(char, "")
	return review


# Remove all punctuation from a single review (. , ? !)
# Replace all punctuation with whitespace
def remove_newlines(review):
	review_word_list = review.split("\n")
	review = " ".join(review_word_list)
	return review


# Remove all links. They usually start with http or https
# This also includes foreign characters
def remove_links_and_emails(review):
	review_word_list = review.split()

	# Remove anything that is a link (usually starts with http or https)
	# Also remove anything with the @ symbol (usually an email)
	# Remove anything that has at least one non-ASCII character
	review_word_list = [word for word in review_word_list if word.isascii() == True and 'http' not in word and '@' not in word and 'ð' not in word and 'Ð' not in word and '€' not in word and 'Ã' not in word and 'Ñ' not in word]

	review = " ".join(review_word_list)
	return review


# Check each sentence for mispellings
# enc is a pyenchant object
# NOT USED - 
def correct_spelling(enc, review):
	review_word_list = review.split()
	review_word_list = [enc.check(word) for word in review_word_list if enc.check(word) == False]

	# Loop through review word list and count the number of mispellings
	# If it is greater than five, list it here
	return len(review_word_list)



# preprocess the reviews to remove non-English and blank reviews.
def preprocess_reviews(all_reviews):
	# Remove all non-English reviews
	print("Removing non-English reviews...")
	non_english_reviews = all_reviews[all_reviews['language'] != 'english']
	all_reviews = all_reviews[all_reviews['language'] == 'english']

	# print(non_english_reviews[['review', 'language']])
	print("Number of reviews after removing non-English reviews: {}\n".format(all_reviews.shape[0]))

	# Remove all blank reviews.
	print("Removing blank reviews...")
	all_reviews = all_reviews[all_reviews['review'] != ""]
	all_reviews = all_reviews[all_reviews['review'] != np.nan]
	all_reviews = all_reviews.dropna(axis='index', subset=['review'])  # Drop NAN reviews
	print("Number of reviews after removing blank reviews: {}".format(all_reviews.shape[0]))

	# Lowercase all text
	print("Lowercasing all reviews...")
	all_reviews['review'] = all_reviews['review'].apply(lambda review: lowercase_text(review))

	# Remove all newlines from the review.
	print("Removing newlines...")
	all_reviews['review'] = all_reviews['review'].apply(lambda review: remove_newlines(review))

	# Remove all punctuation (. , ? ! -)
	# TBD
	print("Removing punctuation...")
	all_reviews['review'] = all_reviews['review'].apply(lambda review: remove_punctuation(review))

	#print("Correct spelling errors...")
	#all_reviews['review'] = all_reviews['review'].apply(lambda review: correct_spelling(review))


	# Remove special characters and links
	# Ex: http, https, @, #, *
	print("Removing special characters...")
	all_reviews['review'] = all_reviews['review'].apply(lambda review: remove_links_and_emails(review))

	# Return the preprocessed data
	return all_reviews


# Split the dataset into the training and test sets.
# reviews is a dataframe with two columns: reviews and voted_up
def split_dataset(reviews):
	return 0


# ====================================================================================================
# MAIN
# ====================================================================================================

def run_model(X_train, X_test):
	# First choice: Use sklearn's Tfidf or CountVectorizer
	print("Fitting vectorizer to training data...")
	# X_train = vectorizer.fit_transform(X_train)

	# Alternative vectorizer: Use Keras Tokenizer instead of sklearn's vectorizers.
	# This will allow us to use the embedding layer in the neural network
	vectorizer = Tokenizer(lower=True)  # Alternative tokenizer for working with the embedding layer
	vectorizer.fit_on_texts(X_train)

	X_train = vectorizer.texts_to_sequences(X_train)  #
	X_test = vectorizer.texts_to_sequences(X_test)  #
	vocab_size = len(
		vectorizer.word_index) + 1  # Comes from the length of the vectorizer's word index. Required for flattening

	X_train = pad_sequences(X_train, padding='post', maxlen=100)
	X_test = pad_sequences(X_test, padding='post', maxlen=100)

	return X_train, X_test, vocab_size


# Main function
def main():
	vader = Vader()

	# Define the autocorrecter
	d = enchant.Dict("en_US")

	# For production - Read all reviews in the directory
	all_reviews = read_all_reviews()

	print("Number of reviews: {}".format(all_reviews.shape[0]))
	print("Finished reading data.\n\n")

	# Separate the dataset into positive and negative reviews.
	data_pos = all_reviews[all_reviews['voted_up'] == True]
	data_neg = all_reviews[all_reviews['voted_up'] == False]

	# # Get a distribution of the data by label
	# plt.title("Label distribution")
	# plt.xlabel("Label")
	# plt.ylabel("Number of reviews (millions)")
	# plt.ticklabel_format(useOffset=False) # Do not show offset with large numbers
	# plt.bar(["Recommended", "Not Recommended"], [data_pos.shape[0], data_neg.shape[0]])
	# plt.savefig("distribution.png")
	# plt.clf()
	# #plt.show()

	# # Preprocess the data.
	# - Remove all non-English reviews (reviews where the value in the language column is not English)
	# NOTE: Some reviews are marked as "english" but have non-English text in them.
	# - Remove all reviews where the review is blank or NaN
	# - Make all reviews lowercase
	print("Preprocessing reviews")
	all_reviews = preprocess_reviews(all_reviews)
	print(all_reviews['review'].head())
	print(all_reviews.shape[0])


	# Separate the reviews and labels from other data
	# Include the requested features
	data = all_reviews[['review', 'voted_up']]

	# # UNUSED CODE
	# # Detect language and remove all reviews where the result is not 'en' (English)
	# data['spelling'] = data['review'].apply(lambda review: correct_spelling(d, review))
	# print(data.head(20))
	# #debug_data = data.head(100)
	# #debug_data.to_excel("spelling_check.xlsx")

	# # Remove rows where the number of mispellings is greater than five
	# data = data[data['spelling'] < 5]
	# print(data.shape[0])

	# Separate the dataset into positive and negative reviews.
	data_pos = data[data['voted_up'] == True]
	data_neg = data[data['voted_up'] == False]

	# # print("Distribution of positive and negative reviews. First number is positive, second number is negative")
	# # print(data_pos.shape[0] / data.shape[0]) # About 87.5% "recommended"
	# # print(data_neg.shape[0] / data.shape[0]) # About 12.4% "not recommended"
	# # print(data_pos.shape[0]) # About 4 million
	# # print(data_neg.shape[0]) # About 570 K

	print("\nSplitting dataset into training and testing")

	# Cut out a lot of positive reviews as the dataset is imbalanced: 4 million reviews are positive, but 570 thousand are negative.
	# Goal: Get about 1 million reviews total with 570 K for training and 570 K for testing.
	data_pos = data_pos.iloc[0:570914, :]  # For all reviews
	data = data_pos.append(data_neg, ignore_index=True)  # Rejoin the negative reviews with the modified positive reviews set.

	# Split the data into the training and test sets.
	# We aim for 400 K reviews (balanced and combined) out of 4.6 million
	X = data['review']
	y = data['voted_up']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=23, stratify=y)

	print("Training with bert")
	temp = []
	for i in y_train:
		if i == True:
			temp.append(int(1))
		else:
			temp.append(int(0))
	bert_history = training_model(X_train, temp)
	print("Training done")


	# Using vader sentiment analysis to get results

	# from nltk.sentiment.vader import SentimentIntensityAnalyzer
	# analyzer = SentimentIntensityAnalyzer()
	# print(analyzer.polarity_scores("story is great but graphic looks like mafia 2 classic"))
	# print(analyzer.polarity_scores("fps wasn't part of our deal."))

	print(type(y_train))
	print("Vader sentiment analysis in progress...")
	vader.vader_analysis(X_train)
	vader_results = vader.vader_validation(y_train)

	# vader_df = pd.DataFrame(vader_results.tolist())  # Convert the list of Vader results into a dataframe.
	print("===========================================================")
	print("Vader prediction accuracy: ", str(round(vader_results * 100, 2)) + "%")
	print("===========================================================")

	# # print(vader_results.tolist())
	# print(vader_df)
	#
	# print(X_train)

	# # Code to get training distribution
	# y_train_pos = y_train[y_train == True]
	# y_train_neg = y_train[y_train == False]

	# y_test_pos = y_test[y_test == True]
	# y_test_neg = y_test[y_test == False]

	# print("y_train distribution")
	# print(y_train_pos.shape[0])  # About 2 million
	# print(y_train_neg.shape[0])  # About 285 K
	# print("\ny_test distribution")
	# print(y_test_pos.shape[0])  # About 2 million
	# print(y_test_neg.shape[0])  # About 285 K
	# ====================================================================================================
	# VECTORIZE THE REVIEWS
	# ====================================================================================================

	try:
		X_train, X_test, vocab_size = run_model(X_train, X_test)
	except:
		exit("Unable to fit vectorizer to training data. Closing program.")
	else:
		print("Successfully fit vectorizer to training data.")
		print("\n")

	# ====================================================================================================
	# CREATE AND TRAIN THE NN
	# ====================================================================================================
	print("Defining the model...")
	NN = define_model(X_train, vocab_size)
	print("\n\n")
	print("Training the model...")
	lstm_history = train_model(NN, X_train, y_train, X_test, y_test, epochs=5)


	# Plot the training accuracy
	epochs = [1, 2, 3, 4, 5]
	plt.plot(epochs, lstm_history.history['accuracy'])
	plt.plot(epochs, bert_history.history['accuracy'])
	plt.title("Training Accuracy of BERT vs LSTM")
	plt.xlabel("Epoch")
	plt.ylabel("Training accuracy")
	plt.legend(['LSTM', 'BERT'], loc='upper left')

	plt.savefig("Steam_SA_training_accuracy.png")

	# Clear the figure and generate the next one
	plt.clf()

	# Plot the training loss
	plt.plot(epochs, lstm_history.history['loss'])
	plt.plot(epochs, bert_history.history['loss'])
	plt.title("Training Loss of BERT vs LSTM")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend(['LSTM', 'BERT'], loc='upper left')

	plt.savefig("Steam_SA_training_loss.png")


# ====================================================================================================
# END OF CODE
# ====================================================================================================

if __name__ == "__main__":
	start = time.time()
	main()
	elapsed = time.time() - start
	print("\n\nScript execution time: {} seconds".format(elapsed))
