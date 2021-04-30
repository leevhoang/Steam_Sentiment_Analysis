# 4/25/2021
# CS 510 - Adventures in NLP
# Steam Sentiment Analysis
#
# 


# Import statements
import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to all CSVs
DATA_PATH = 'game_rvw_csvs'

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
	print("Total number of rows")
	print(length)
	return all_reviews


# Main function
def main():
	# data = pd.read_csv(DATA_PATH + "/" + "10_CounterStrike.csv", index_col=False, header=0)

	# # reviews = data[['review', 'voted_up']]

	# # print(data.head())
	# # print(reviews)
	print("Reading all game reviews now:")
	all_reviews = read_all_reviews()
	print(all_reviews)
	print(all_reviews.shape[0])

	print("Finished reading data.\n\n")
	# Preprocess the data.
	# - Remove all non-English reviews
	# - Remove all reviews where the review is blank
	print("Preprocessing reviews")
	non_english_reviews = all_reviews[all_reviews['language'] != 'english']
	all_reviews = all_reviews[all_reviews['language'] == 'english']

	# One of the reviews that isn't marked 'english' actually has an English language review.
	all_reviews.to_csv('all_reviews.csv')
	non_english_reviews.to_csv('non_english_reviews.csv', index=False)

	print(non_english_reviews[['review', 'language']])
	print(all_reviews.shape[0])


	data = all_reviews[['review', 'voted_up']]
	print(data.head())

	data_pos = data[data['voted_up'] == True]
	data_neg = data[data['voted_up'] == False]

	print("Distribution of positive and negative reviews. First number is positive, second number is negative")
	print(data_pos.shape[0] / data.shape[0]) # About 87.5% "recommended"
	print(data_neg.shape[0] / data.shape[0]) # About 12.4% "not recommended"

main()