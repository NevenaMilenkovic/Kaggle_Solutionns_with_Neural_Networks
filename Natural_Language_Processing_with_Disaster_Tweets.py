# Accuracy 81.2%

import numpy as np 
import pandas as pd 
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, Embedding, Model
from sklearn.metrics import confusion_matrix


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

train_df['text']=train_df['text'].str.replace(r'[a-zA-Z-1-9]*\n[a-zA-Z-1-9]*','')
print(train_df['text'][10:20])
train_df['text']=train_df['text'].str.replace('[{}]'.format(string.punctuation), '', regex=True)
print(train_df['text'][10:20])
train_df['text']=[word.encode('ascii',errors='ignore') for word in train_df['text']]

train_df['text']=[word.decode('utf-8',errors='ignore') for word in train_df['text']]
train_df['text']=train_df['text'].str.replace(r'\bhttp[a-zA-Z-1-9]*','')
train_df['text']=train_df['text'].str.replace(r'\w*\d\w*','')
train_df['text']=train_df['text'].str.strip()
train_df['text']=train_df['text'].str.replace('  ',' ')
train_df['text']=train_df['text'].str.lower()

stop = stopwords.words('english')
train_df['text']=train_df["text"].apply(lambda words: ' '.join(word for word in words.split() if word not in stop))

max_vocab_size=20000
tokenizer=Tokenizer()
tokenizer.fit_on_texts(train_df['text'])
sequences_train=tokenizer.texts_to_sequences(train_df['text'])
sequences_test=tokenizer.texts_to_sequences(test_df['text'])
word2idx=tokenizer.word_index
V=len(word2idx)

data_train=pad_sequences(sequences_train)
T=data_train.shape[1]

data_test=pad_sequences(sequences_test, maxlen=T)

path_to_glove_file = '/kaggle/input/glove-twitter/glove.twitter.27B.25d.txt'

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

num_tokens = V + 2
embedding_dim = 25
hits = 0
misses = 0
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    trainable=True,
)
embedding_layer.build((1,))
embedding_layer.set_weights([embedding_matrix])

optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)

int_sequences_input=tf.keras.layers.Input(shape=(None,))
embedded_sequences=embedding_layer(int_sequences_input)
x=tf.keras.layers.LSTM(10,return_sequences=True)(embedded_sequences)
x=tf.keras.layers.GlobalMaxPooling1D()(x)
preds=tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(int_sequences_input, preds)

model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['binary_accuracy'])

history=model.fit(data_train, train_df["target"].values, validation_split=0.2,epochs=7)
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss','val_loss']].plot()
history_frame.loc[:, ['binary_accuracy','val_binary_accuracy']].plot()

y_pred = model.predict(data_train)
y_pred = [1 if a>=0.5 else 0 for a in y_pred]
cm_dot_five = confusion_matrix(train_df['target'].values, y_pred)
print(cm_dot_five)

model.fit(data_train, train_df["target"].values,epochs=7)

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
p= model.predict(data_test)
sample_submission["target"] = [1 if a>=0.5 else 0 for a in p]
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)
