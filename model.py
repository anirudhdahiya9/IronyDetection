#import keras
from keras.layers import Input, LSTM, Dense, Embedding, MaxPooling1D, Flatten, Conv1D, Bidirectional
from keras.models import Sequential, load_model
#import codecs
#from nltk import word_tokenize
import pandas as pd
import csv
import numpy as np
#from keras.preprocessing.sequence import pad_sequences
import pickle

embedding_size = 50

class LSTMModel:
    def __init__(self, vocab_size, embmatrix):
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, output_dim=embedding_size, weights=[embmatrix]))
        self.model.add(Bidirectional(LSTM(128)))
        #model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    def train(self, data, labels):
        self.model.fit(data, labels, epochs=1)

    def evaluate(self, data, labels):
        score = self.model.evaluate(data, labels)
        return score

    def save(self):
        self.model.save('LSTMmodel.h5')

class CNNModel:
    def __init__(self, vocab_size, embmatrix):
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, output_dim=embedding_size, weights=[embmatrix], input_length=40))

        self.model.add(Conv1D(32, 4, activation='relu'))
        #self.model.add(Conv1D(8, 2, activation='relu'))

        self.model.add(MaxPooling1D(4))
        self.model.add(Flatten())

        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def train(self, data, labels):
        self.model.fit(data, labels, epochs=1)

    def evaluate(self, data, labels):
        score = self.model.evaluate(data, labels)
        return score

    def save(self):
        self.model.save('CNNmodel.h5')


def loadData():
    data, labels = pickle.load(open('trainingData.pkl'))[0]
    return data, labels

def prepvocab():
    #LOAD EMBEDDINGS
    words = pd.read_table('glove.6B.'+str(embedding_size)+'d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    words_matrix = words.as_matrix()

    #Add vectors for <UNK>, <EOS> and <PAD>
    bla = np.zeros((2, embedding_size))
    words_matrix = np.concatenate((bla, words_matrix), axis=0)

    vocab = ['<pad>', '<unk>'] + list(words.index)
    vocab_size = len(vocab)
    dictionary = dict(zip(vocab, range(vocab_size)))
    
    return words_matrix, dictionary

words_matrix, dictionary = prepvocab()
data, labels = loadData()

print data[:5]
print labels[:5]
train_data, tr_labels = (data[:3500], labels[:3500])
test_data, te_labels = (data[3500:], labels[3500:])

'''
model = load_model('LSTMmodel.h5')
predictions = model.predict(test_data)

fp=0
fn=0
tp=0
tn=0
for ind, pred in enumerate(predictions):
    if pred[0]<0.5:
        if int(te_labels)==0:
            tn+=1
        else:
            fn+=1
    else:
        if int(te_labels)==1:
            tp+=1
        else:
            fp+=1
'''


'''
model = LSTMModel(len(dictionary.keys())+2, words_matrix)
for i in range(15):
    model.train(train_data, tr_labels)
    print model.evaluate(test_data, te_labels)
model.save()


'''
cnnmodel = CNNModel(len(dictionary.keys())+2 ,words_matrix)
for i in range(15):
    cnnmodel.train(train_data, tr_labels)
    print cnnmodel.evaluate(test_data, te_labels)
cnnmodel.save()
