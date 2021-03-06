import math
import random
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert

BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 2
DROPOUT_RATE = 0.2
NB_EPOCHS = 5


class TEXT_MODEL(tf.keras.Model):
	def __init__(self, vocabulary_size, embedding_dimensions=128, cnn_filters=50, dnn_units=512, model_output_classes=2,
	             dropout_rate=0.1, training=False, name="text_model"):
		super(TEXT_MODEL, self).__init__(name=name)

		self.embedding = layers.Embedding(vocabulary_size, embedding_dimensions)
		self.cnn_layer1 = layers.Conv1D(filters=cnn_filters, kernel_size=2, padding="valid", activation="relu")
		self.pool = layers.GlobalMaxPool1D()

		self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
		self.dropout = layers.Dropout(rate=dropout_rate)
		if model_output_classes == 2:
			self.last_dense = layers.Dense(units=1, activation="sigmoid")
		else:
			self.last_dense = layers.Dense(units=model_output_classes, activation="softmax")

	def call(self, inputs, training):
		l_0 = self.embedding(inputs)
		l_1 = self.cnn_layer1(l_0)
		l_1 = self.pool(l_1)

		concatenated = tf.concat([l_1], axis=-1)
		concatenated = self.dense_1(concatenated)
		concatenated = self.dropout(concatenated, training)
		model_output = self.last_dense(concatenated)

		return model_output


def tokenize_reviews(text_reviews):
	return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))


def training_model(reviews_series, y_train, BATCH_SIZE=32):
	reviews = reviews_series.tolist()
	print("tokenizing...")
	tokenized_result = [tokenize_reviews(review) for review in reviews]
	print("merging labels...")
	reviews_with_len = [[review, y_train[i], len(review)] for i, review in enumerate(tokenized_result)]

	print("sorting reviews...")
	random.shuffle(reviews_with_len)

	sorted_reviews_labels = [(review_lab[0], review_lab[1]) for review_lab in reviews_with_len]

	processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels, output_types=(tf.int32, tf.int32))

	batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None,), ()))

	TOTAL_BATCHES = math.ceil(len(sorted_reviews_labels) / BATCH_SIZE)
	TEST_BATCHES = TOTAL_BATCHES // 10
	batched_dataset.shuffle(TOTAL_BATCHES)
	test_data = batched_dataset.take(TEST_BATCHES)
	train_data = batched_dataset.skip(TEST_BATCHES)

	text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH, embedding_dimensions=EMB_DIM, cnn_filters=CNN_FILTERS,
	                        dnn_units=DNN_UNITS, model_output_classes=OUTPUT_CLASSES, dropout_rate=DROPOUT_RATE)

	text_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	print("Bert training progress:")
	history = text_model.fit(train_data, epochs=NB_EPOCHS)

	# Evaluate BERT on the test set
	print("Evaluating Bert's training result... ")
	results = text_model.evaluate(test_data)
	print(results)

	# Return the history to plot accuracy and loss
	return history
