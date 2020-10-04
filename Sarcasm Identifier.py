#This is a simple NLP python code, and its purpose is to understand that whether an input sentence is sarcastic or not.

import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from  tensorflow.keras.preprocessing.sequence import pad_sequences

#Initial Parameters

trainingSize = 20000
vocabSize = 1000
maxLength = 40
paddingType = 'post'
truncatingType = 'post'
numEpochs= 30
embeddingDim = 16

#downloading the training data and importing it to our program

dataSet = pd.read_json('sarcasm.json', lines= True)

headlines = []
labels = []
urls = []

headlines = dataSet['headline'].to_list()
labels = dataSet['is_sarcastic'].to_list()
urls = dataSet['article_link'].to_list()

#Data Preprocessing

training_sentences= headlines[0:trainingSize]
testing_sentences= headlines[trainingSize:]
training_labels= labels[0:trainingSize]
testing_labels= labels[trainingSize:]

#Using the tokenizer in order to allocate a number to each word

tokenizer= Tokenizer(num_words= vocabSize, oov_token= "<OOV>")
tokenizer.fit_on_texts(training_sentences)

word_index= tokenizer.word_index

training_sequences= tokenizer.texts_to_sequences(training_sentences)
training_padded= pad_sequences(training_sequences, maxlen= maxLength, padding= paddingType, truncating= truncatingType)

testing_sequences= tokenizer.texts_to_sequences(testing_sentences)
testing_padded= pad_sequences(testing_sequences, maxlen= maxLength, padding= paddingType, truncating= truncatingType)

#changing training and testing data into numpy arrays

training_x= np.asarray(training_padded)
training_y= np.asarray(training_labels)
validation_x= np.asarray(testing_padded)
validation_y= np.asarray(testing_labels)

#Instantiating the model and training it

model= tf.keras.Sequential([
                             tf.keras.layers.Embedding(vocabSize, embeddingDim, input_length= maxLength),
                             tf.keras.layers.GlobalAveragePooling1D(),
                             tf.keras.layers.Dense(24, activation= 'relu'),
                             tf.keras.layers.Dense(1, activation= 'sigmoid')
                             ])

model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['accuracy'])


history= model.fit(training_x, training_y, epochs= numEpochs, validation_data= (validation_x, validation_y), verbose= 2)


#Testing our model with a new input sentenc

inputSentence= ['I want to learn more']
sequences= tokenizer.texts_to_sequences(inputSentence)
padded= pad_sequences(sequences, maxlen = maxLength, padding= paddingType, truncating= truncatingType)


print(model.predict(np.asarray(padded)))
