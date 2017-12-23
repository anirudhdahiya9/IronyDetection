from nltk.tokenize import TweetTokenizer
import pandas as pd
import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import re
from split_upperhashtags import split_hashtag
from split_hashtags import sp
import pickle

tknzr = TweetTokenizer()

url_pattern = re.compile('http[^\s]+', flags=re.UNICODE)


trainingpath = '/Users/anirudhdahiya/courses/IS/model/data/data/train/SemEval2018-T4-train-taskA.txt'
#testingpath = '/Users/anirudhdahiya/courses/IS/model/data/data/test/SemEval2018-T4-test-taskA.txt'

def loadTokenize(path):
    with open(path) as f:
        lines = f.readlines()[1:]
        for iline, lin in enumerate(lines):
            lines[iline] = lin[:-1].split('\t')
            try:
                lines[iline][2] = url_pattern.sub(r'', lines[iline][2])
            except:
                continue
            lines[iline][2] = tknzr.tokenize(lines[iline][2])

    return lines

def splittags(data):

    for ind, line in enumerate(data):
        print ind
        indices = []
        newtags = []
        for itk, tk in enumerate(line[2]):
            if tk.startswith('#'):
                if len(tk)>1 and tk[1].isupper():
                    #print '>>', split_hashtag(tk[1:])
                    newtags += [split_hashtag(tk[1:])]
                    indices += [itk]
                else:
                    tempsplit = sp(tk[1:].lower())
                    print '~~', tempsplit
                    if type(tempsplit[0])==type([]):
                        newtags+=[[' '.join(tempsplit[0])]]
                    else:
                        newtags += [tempsplit]

                    indices += [itk]

        #print newtags

        ptr1 = 0
        ptr2 = 0
        final = []
        while ptr1<len(line[2]):
            if ptr2<len(indices) and indices[ptr2]==ptr1:
                final += ['#'] + newtags[ptr2]
                ptr2+=1
                ptr1+=1
            else:
                final+=[line[2][ptr1]]
                ptr1 += 1

        data[ind][2] = final
        print final
    return data

def prepvocab():
    # LOAD EMBEDDINGS
    embedding_size = 50

    words = pd.read_table('glove.6B.' + str(embedding_size) + 'd.txt', sep=" ", index_col=0, header=None,
                          quoting=csv.QUOTE_NONE)
    words_matrix = words.as_matrix()

    # Add vectors for <UNK>, <EOS> and <PAD>
    bla = np.zeros((2, embedding_size))
    words_matrix = np.concatenate((bla, words_matrix), axis=0)

    vocab = ['<pad>', '<unk>'] + list(words.index)
    vocab_size = len(vocab)
    dictionary = dict(zip(vocab, range(vocab_size)))

    return words_matrix, dictionary

def tkids(data, dictionary):
    for ind, line in enumerate(data):
        for i in range(len(line[2])):
            try:
                line[2][i] = dictionary[line[2][i].lower()]
            except:
                line[2][i] = dictionary['<unk>']
        data[ind][2] = line[2]
    return data

def sepNpad(data):
    data_a = np.array([i[2] for i in data])
    #print data_a.shape
    labels = [i[1] for i in data]
    data_a = pad_sequences(data_a, maxlen=40, dtype='int32', padding='pre', truncating='pre', value=0)
    return data_a, labels

def prepData(path):
    print 'Prep Data', path
    data = loadTokenize(path)
    print 'tokenized'
    data = splittags(data)
    print 'tags split'
    data = tkids(data, dictionary)
    print 'tkids converted'
    data_a, labels = sepNpad(data)
    return data_a , labels

words_matrix, dictionary = prepvocab()
pickle.dump([prepData(trainingpath)], open('trainingData.pkl', 'w'))
#pickle.dump([prepData(testingpath)], open('testData.pkl', 'w'))