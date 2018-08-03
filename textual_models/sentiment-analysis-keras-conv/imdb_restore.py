import keras
from keras.preprocessing import sequence

NUM_WORDS=10000 # only use top 1000 words
INDEX_FROM=3   # word index offset

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS) # index_from=INDEX_FROM
X_train, y_train = train
X_test,  y_test  = test

max_review_length = 30
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
sentence = X_train[0]
sentiment = y_train[0]
print("Sentiment:")
print(sentiment)
print("Sentence:")
print(' '.join(id_to_word[id] for id in sentence ))
