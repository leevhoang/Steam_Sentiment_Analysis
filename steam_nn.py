# 5/10/2021
# CS 510 - Adventures in NLP
# Steam Sentiment Analysis
#
# This file contains all of the primary neural network code.
# WORK IN PROGRESS
#
# You will need to run "pip install tensorflow" to be able to execute this code.

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
def define_model():

	# Set up an empty model and the first three convolutional layers.
	# Each convolutional layer has an activation function and
	# a max pooling layer.
	#
	# The final layer has a sigmoid activation function meant for storing
	# the prediction.
	# =============================================================================
	model = tf.keras.Sequential([
		# INPUT LAYER
		Embedding(1000, 32, input_length = 500000),

		# HIDDEN LAYERS and OTHER BLOCKS

		# OUTPUT LAYER
		Dense(1, activation='sigmoid')
	])
	# =============================================================================

	# Compile the finished model
	# Binary problems like this one require a binary cross entropy loss
	model.compile(
		loss='binary_crossentropy',
		optimizer='rmsprop',
		metrics=['accuracy'] # We will measure the accuracy.
		)

	# Return the finished model
	return model


# Placeholder function for training the model
def train_model(model, X_train, y_train):
	#dataset = tf.sparse.reorder(X_train)
	X_train = X_train.values
	#X_train = np.reshape(X_train, (-1, X_train.shape[0]))
	model.fit(X_train, y_train, epochs=10)
	# try:
	# 	print("\nTraining the model...")
	# 	dataset = tf.sparse.reorder(X_train)
	# 	model.fit(X_train, y_train, epochs=10)
	# except Exception as e:
	# 	print("ERROR - Unable to train the model - closing program. Please see the error below for more details.")
	# 	print(e)
	# 	print("\nProgram closed with error\n")
	# 	exit()
	#return 0