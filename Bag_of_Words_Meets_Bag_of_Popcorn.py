# Imdb dataset Accuracy 90%

import pandas as pd
import numpy as np
train=pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip',header=0,delimiter='\t',quoting=3)
train.head()

from bs4 import BeautifulSoup
import nltk
nltk.download('all')
from nltk.corpus import stopwords

def review_to_words(raw_review):
    review_text=BeautifulSoup(raw_review).get_text()     
    letters_only=re.sub('[^a-zA-Z]',' ',review_text)
    words=letters_only.lower().split()
    stops = set(stopwords.words("english"))                  
    throw=['movie','some','that']
    meaningful_words=[w for w in words if not w in stops]
    meaningful_words=[w for w in words if not w in throw]
    return(' '.join(meaningful_words))

num_reviews=train['review'].size
clean_train_reviews=[]
for i in range(0, num_reviews):
    if( (i+1)%1000 == 0 ):
        print("Review %d of %d\n" % ( i+1, num_reviews ))    
    clean_train_reviews.append(review_to_words(train['review'][i]))

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features=15000,min_df=5,ngram_range=(1,2))
train_data_features=vectorizer.fit_transform(clean_train_reviews)
train_data_features=train_data_features.toarray()

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

model=tf.keras.Sequential([
                        
                          layers.Dense(4,activation='swish'),
                          layers.Dropout(0.2),
                          layers.Dense(1,activation='sigmoid')])

from tensorflow.keras.optimizers import Adam
optimizer=Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,loss='binary_crossentropy',
              metrics=['binary_accuracy'])

history = model.fit(train_data_features, train['sentiment'],epochs=7,verbose=0,validation_split=0.2)
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss','val_loss']].plot()
history_frame.loc[:, ['binary_accuracy','val_binary_accuracy']].plot()
print(history_frame.loc[:, ['val_loss']])
print(history_frame.loc[:, ['val_binary_accuracy']])

history = model.fit(train_data_features, train['sentiment'],epochs=7,verbose=0)

test=pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip',header=0,delimiter='\t',quoting=3)
num_reviews=len(test['review'])
clean_test_reviews=[]
for i in range(0, num_reviews):
    if ((i+1)%1000==0):
        print('Review %d of %d\n'%(i+1,num_reviews))
    clean_review=review_to_words(test['review'][i])
    clean_test_reviews.append(clean_review)
test_data_features=vectorizer.transform(clean_test_reviews)
test_data_features=test_data_features.toarray()
result=model.predict(test_data_features)
result = [1 if a>=0.5 else 0 for a in result]
print(result[:10])
output=pd.DataFrame(data={'id':test['id'],'sentiment':result})
output.to_csv('Bag_of_Words_model.csv',index=False,quoting=3)
