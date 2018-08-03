# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Convolution2D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard

def imdb_restore(sentence):
	INDEX_FROM = 3
	word_to_id = imdb.get_word_index()
	word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
	word_to_id["<PAD>"] = 0
	word_to_id["<START>"] = 1
	word_to_id["<UNK>"] = 2

	id_to_word = {value:key for key,value in word_to_id.items()}
	sent_restored = ' '.join(id_to_word[id] for id in sentence)

	#print("Sentence:")
	#print(sent_restored)
	return sent_restored

def save_model(file_prefix, model):
	model_json_filename = '{}.json'.format(file_prefix)
	model_h5_filename = '{}.h5'.format(file_prefix)

	# serialize model to JSON
	model_json = model.to_json()
	with open(model_json_filename, "w") as json_file:
		json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights(model_h5_filename)
	print("Saved model and weights to disk.")
	print("Model is stored in {} file.".format(model_json_filename))
	print("Weights are stored in {} file in HDF5 format.".format(model_h5_filename))

# Using keras to load the dataset with the top_words
top_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# Pad the sequence to the same length
max_review_length = 30
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Using embedding from Keras
embedding_vector_length = 30
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='valid'))
model.add(Convolution1D(32, 3, padding='valid'))
model.add(Convolution1D(16, 3, padding='valid'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, callbacks=[tensorBoardCallback], batch_size=64)

# Evaluation on the test set
scores = model.evaluate(X_test, y_test, verbose=0)
accuracy = scores[1]*100
print("Accuracy: %.2f%%" % (scores[1]*100))

#print(X_test[0])
#print(type(X_test[0]))
#print(X_test[0].shape)
ind = 2
prediction = model.predict(X_test[ind:ind+1])
label      = y_test[ind]
coded      = X_test[ind]
sentence   = imdb_restore(X_test[ind])
print("Prediction:{}, Actual Label:{}".format(prediction, label))
print("Sentence: {}".format(sentence))

#Save the model
model_temp = 'model-len{}-embed{}'.format(max_review_length,embedding_vector_length)
save_model(model_temp, model)



#Save the sentence for symbolic analysis
example_filename = 'examples.txt'
with open(example_filename, "w") as ex_file:
	ex_file.write(str(label))
	for val in coded:
		ex_file.write(',{}'.format(val))

with open('example_sent.txt','w') as ex_file:
	ex_file.write(str(label))
	for word in sentence.split():
		ex_file.write(',{}'.format(word))
