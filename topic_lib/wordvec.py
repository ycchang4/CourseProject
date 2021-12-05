from gensim.test.utils import datapath
from gensim import utils

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self):
        p = []
        with open("phrases.txt") as f:
            for line in f:
                p.append(line.split())
        self.p = dict(p)
    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            s = utils.simple_preprocess(line)
            i = 0
            new_s = []
            while i < len(s) - 1:
                if (s[i+1] == self.p.get(s[i], False)):
                    new_s.append(s[i] + "_" + s[i+1]) 
                    i+=2
                else:
                    new_s.append(s[i])
                    i+=1
            yield new_s


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class moreCorpus:
    def __iter__(self):
        for line in open("phrased_bags.dat"):
            yield line.split()

import gensim.models

sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)

text8 = gensim.models.word2vec.Text8Corpus("text8")
model.build_vocab(text8, update=True)
model.train(text8, total_examples=model.corpus_count, epochs=model.epochs)
more_sentences = moreCorpus()
model.build_vocab(more_sentences, update=True)
model.train(more_sentences, total_examples=model.corpus_count, epochs=model.epochs)

model.wv.most_similar('search_engine')

model.wv.similarity("test", "analyze")

import numpy as np
import tensorflow as tf
from tensorflow import keras

hitx = []
hity = []
pos = 0
neg = 0


with open("miss.dat") as f:
    for line in f:
        for word in utils.simple_preprocess(line):
            try:
                hitx.append(model.wv[word])
                hity.append(0.0)
                neg += 1
            except:
                continue
with open("hits.dat") as f:
    for line in f:
        for word in utils.simple_preprocess(line):
            try:
                vec = model.wv[word]
                hitx.append(vec)
                hity.append(1.0)
                pos += 1
            except:
                continue

hitx = np.array(hitx)
hity = np.array(hity)