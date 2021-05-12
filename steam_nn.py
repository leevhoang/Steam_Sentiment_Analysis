# 5/10/2021
# CS 510 - Adventures in NLP
# Steam Sentiment Analysis
#
# This file contains all of the primary neural network code.
# WORK IN PROGRESS
#

# Import statements for tensorflow.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Embedding


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

	# INPUT LAYER

	# HIDDEN LAYERS and OTHER BLOCKS

	# OUTPUT LAYER

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