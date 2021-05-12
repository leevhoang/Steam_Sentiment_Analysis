# 4/25/2021
# CS 510 - Adventures in NLP
# Steam Sentiment Analysis
#
# This file reads and preprocesses reviews from the Steam Game Reviews dataset.
# The dataset can be found here: https://www.kaggle.com/smeeeow/steam-game-reviews
# To use this file, download the dataset from the above link and extract its contents to the same directory as main_app.py.


# Import statements
import pandas as pd # For reading CSV files
import matplotlib.pyplot as plt # For making plots
import os
import time # For timing how long it takes to run the script.

# SKlearn - For splitting the dataset into the training and testing sets.
from sklearn.model_selection import train_test_split

# Import neural network code from a separate python file
# 5/10/2021
from steam_nn import define_model

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


	for reviews in os.listdir(DATA_PATH):
		print("Reading " + reviews)
		data = pd.read_csv(DATA_PATH + "/" + reviews)
		length += data.shape[0]
		all_reviews.append(data) 

	all_reviews = pd.concat(all_reviews, axis=0, ignore_index=True)
	print("Total number of rows: {}".format(length))
	return all_reviews

# ====================================================================================================
# PREPROCESSING
# ====================================================================================================

# preprocess the reviews to remove non-English and blank reviews.
def preprocess_reviews(all_reviews):
	# Remove all non-English reviews
	print("Removing non-English reviews")
	non_english_reviews = all_reviews[all_reviews['language'] != 'english']
	all_reviews = all_reviews[all_reviews['language'] == 'english']

	#print(non_english_reviews[['review', 'language']])
	print("Number of reviews after removing non-English reviews: {}\n".format(all_reviews.shape[0]))

	# Remove all blank reviews.
	print("Removing blank reviews")
	all_reviews = all_reviews[all_reviews['review'] != ""]
	print("Number of reviews after removing blank reviews: {}".format(all_reviews.shape[0]))

	# Return the preprocessed data
	return all_reviews

# Split the dataset into the training and test sets.
# reviews is a dataframe with two columns: reviews and voted_up
def split_dataset(reviews):
	return 0

# ====================================================================================================
# MAIN
# ====================================================================================================


# Main function
def main():
	#print("Reading all game reviews now:")
	all_reviews = read_all_reviews()
	#print(all_reviews)
	print("Number of reviews: {}".format(all_reviews.shape[0]))

	print("Finished reading data.\n\n")

	# Preprocess the data.
	# - Remove all non-English reviews
	# - Remove all reviews where the review is blank
	print("Preprocessing reviews")
	all_reviews = preprocess_reviews(all_reviews)

	# Separate the reviews and labels from other data
	data = all_reviews[['review', 'voted_up']]

	# Separate the dataset into positive and negative reviews.
	data_pos = data[data['voted_up'] == True]
	data_neg = data[data['voted_up'] == False]

	# print("Distribution of positive and negative reviews. First number is positive, second number is negative")
	# print(data_pos.shape[0] / data.shape[0]) # About 87.5% "recommended"
	# print(data_neg.shape[0] / data.shape[0]) # About 12.4% "not recommended"
	# print(data_pos.shape[0]) # About 4 million
	# print(data_neg.shape[0]) # About 570 K

	print("\nSplitting dataset into training and testing")

	# Cut out a lot of positive reviews as the dataset is imbalanced: 4 million reviews are positive, but 570 thousand are negative.
	# Goal: Get about 1 million reviews total with 570 K for training and 570 K for testing.
	data_pos = data_pos.iloc[0:570914, :]
	data = data_pos.append(data_neg, ignore_index=True) # Rejoin the negative reviews with the modified positive reviews set.

	# Split the data into the training and test sets.
	# We aim for 400 K reviews (balanced and combined) out of 4.6 million
	X = data['review']
	y = data['voted_up']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=23, stratify=y)

	y_train_pos = y_train[y_train == True]
	y_train_neg = y_train[y_train == False]

	y_test_pos = y_test[y_test == True]
	y_test_neg = y_test[y_test == False]

	print("y_train distribution")
	print(y_train_pos.shape[0]) # About 2 million
	print(y_train_neg.shape[0]) # About 285 K
	print("\ny_test distribution")
	print(y_test_pos.shape[0]) # About 2 million
	print(y_test_neg.shape[0]) # About 285 K


# ====================================================================================================
# END OF CODE
# ====================================================================================================


# Main function
start = time.time()
main()
elapsed = time.time() - start
print("\n\nScript execution time: {} seconds".format(elapsed))