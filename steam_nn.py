# 5/10/2021
# CS 510 - Adventures in NLP
# Steam Sentiment Analysis
#
# This file contains all of the primary neural network code.
# WORK IN PROGRESS
#
# You will need to install tensorflow or tensorflow-gpu to be able to execute this code.
# However, it is recommended to install tensorflow-gpu for faster training times.

# Import statements for tensorflow.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Embedding

from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np



# Target size for neural network input.
# To be adjusted later
target_size = (1, 1)


# Define the neural network, which will take as input a review.
# It will output a prediction.
def define_model(X_train, vocab_size):

	# Set up an empty model and the first three convolutional layers.
	# Each convolutional layer has an activation function and
	# a max pooling layer.
	#
	# The final layer has a sigmoid activation function meant for storing
	# the prediction.
	# =============================================================================
	print("VOCAB SIZE {}".format(vocab_size))
	print("\n\n\n")
	model = tf.keras.Sequential([
		# INPUT LAYER
		Embedding(input_dim=vocab_size, output_dim=64, input_length=100), # For using word embeddings. Right now, using CountVectorizer on input will cause the code to crash with the embedding layer.
		#Flatten(),
		# HIDDEN LAYERS and OTHER BLOCKS
		LSTM(15, dropout=0.5), # LSTM layer

		# OUTPUT LAYER
		# Uses a sigmoid activation function because we are doing binary classification
		Dense(1, activation='sigmoid')
	])
	# =============================================================================

	# Compile the finished model
	# Binary problems like this one require a binary cross entropy loss
	model.compile(
		loss='binary_crossentropy', 
		optimizer='rmsprop', # Default optimizer. 
		metrics=['accuracy'] # We will measure the accuracy.
		)

	# Return the finished model
	print(model.summary())
	return model


# Placeholder function for training the model
def train_model(model, X_train, y_train, X_test, y_test, epochs):
	try:
		print("\nTraining the model...")
		history = model.fit(x=X_train, y=y_train, epochs=epochs) # x expects a numpy array or a list of arrays. y should be the same time as x
	except Exception as e:
		print("ERROR - Unable to train the model - closing program. Please see the error below for more details.")
		print(e)
		print("\nProgram closed with error\n")
		exit()
	else:
		print("Training successful. Evaluating the model on the test set...")
		loss, accuracy = model.evaluate(X_test, y_test)
		print("TESTING ACCURACY: {}".format(accuracy))