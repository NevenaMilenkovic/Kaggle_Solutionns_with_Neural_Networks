#MNIST dataset, accuracy 98.3%

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


xtrain=pd.read_csv('/kaggle/input/digit-recognizer/train.csv',dtype=float)
df_test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv',dtype=float)

ytrain=xtrain['label'].values
xtrain=xtrain.drop(columns='label').values

xtest=df_test.copy().values

xtrain=xtrain/255
xtest=xtest/255
xtrain = xtrain.reshape(xtrain.shape[0], 28, 28,1)
input_shape = (28, 28, 1)
xtest = xtest.reshape(xtest.shape[0], 28, 28,1) 


from tensorflow.keras.utils import to_categorical
ytrain = keras.utils.to_categorical(ytrain, 10)

model=tf.keras.Sequential([
                          layers.Flatten(input_shape=(28,28)),
                       
                          layers.Dense(512,activation='swish'),
                          #layers.Dropout(0.3),
                          layers.Dense(512,activation='swish'),
                          #layers.Dropout(0.1),
                          #layers.Dense(512,activation='swish'),
                          #layers.Dense(512,activation='swish'),
                          #layers.Dropout(0.1),
                          layers.Dense(10,activation='softmax')])

from tensorflow.keras.optimizers import Adam
optimizer=Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(xtrain,ytrain,epochs=150,verbose=0)
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss']].plot()
history_frame.loc[:, ['accuracy']].plot();

prediction=model.predict(xtest)

import csv
num_classes = prediction.shape[1]

with open('submission.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['ImageId', 'Label'])

    for i, pred in enumerate(prediction):
        label = pred.argmax()

        writer.writerow([i+1, label])
