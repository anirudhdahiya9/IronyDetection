import nltk
from nltk.corpus import words, brown

word_dictionary = list(set(words.words()))

for alphabet in "bcdefghjklmnopqrstuvwxyz":
    word_dictionary.remove(alphabet)

def split_hashtag(hashtag):
    breaks = []
    fl=0
    for i in hashtag:
        if not i.isupper():
            fl=1
            break
    if fl==0:
        return [hashtag]

    if hashtag[0].isupper():
        for i, ch in enumerate(hashtag):
            if ch.isupper():
                breaks += [i]

        final = []
        for i in range(len(breaks)-1):
            final += [hashtag[breaks[i]:breaks[i+1]]]
        final += [hashtag[breaks[-1]:]]

        return final
    return []