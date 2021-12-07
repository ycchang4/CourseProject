from gensim.test.utils import datapath
from gensim import utils
import gensim.models

import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings

class WordVecFilter:
    def __init__(self):
        
        self.model=None
        self.corpus_list = []
        self.hitx = []
        self.hity = []
        self.nnet = None
        self.pos = 0
        self.neg = 0
    class PhrasedCorpus:
        """An iterator that yields sentences (lists of str)."""
        def __init__(self, path=None):
            p = []
            self.path = path
            with open("phrases.txt", encoding="utf-8") as f:
                for line in f:
                    phrase = line.split()
                    prefix = phrase[0]
                    for x in phrase[1:]:
                        p.append((prefix, x))
                        prefix += "_" + x
            self.p = dict(p)
        def __iter__(self):
            if (self.path == None):
                corpus_path = datapath('lee_background.cor')
            else:
                corpus_path = self.path
            for line in open(corpus_path, encoding="utf-8"):
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
                nns =[]
                i = 0
                while i < len(new_s) - 1:
                    if (new_s[i+1] == self.p.get(new_s[i], False)):
                        nns.append(new_s[i] + "_" + new_s[i+1])
                        i+=2
                    else:
                        nns.append(new_s[i])
                        i+=1
                yield nns


    class BasicCorpus:
        def __init__(self, path):
            self.path = path
        def __iter__(self):
            for line in open(self.path, encoding="utf-8"):
                yield line.split()
    class FastGetter:
        def __init__(self, m, wv):
            self.model = m
            self.wv = wv
            self.d = {}
        def get(self, x):
            r = self.d.get(x, 100)
            if (r != 100):
                return r
            else:
                r = self.model.predict(np.array([self.wv[x.strip()]]))
                self.d[x] = r
                return r

    def add_basic_corpus(self, path):
        self.corpus_list.append(self.BasicCorpus(path))
        return len(self.corpus_list) - 1
    def add_phrased_corpus(self, path):
        self.corpus_list.append(self.PhrasedCorpus(path))
        return len(self.corpus_list) - 1
    def train_corpus(self, cid):
        if self.model == None:
            self.model = gensim.models.Word2Vec(sentences=self.corpus_list[cid])
        else:
            self.model.build_vocab(self.corpus_list[cid], update=True)
            self.model.train(self.corpus_list[cid], total_examples=self.model.corpus_count, epochs=self.model.epochs)
    def build_nnet(self):
        initial_bias = np.log([self.pos/self.neg])
        output_bias = tf.keras.initializers.Constant(initial_bias)
        self.nnet = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='tanh'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid',bias_initializer= output_bias)])
        self.nnet.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

    def add_to_miss(self,path):
        if (self.nnet != None):
            warnings.warn("Adding data after initialization of nnet")
        with open(path) as f:
            for line in f:
                for word in utils.simple_preprocess(line):
                    try:
                        self.hitx.append(self.model.wv[word])
                        self.hity.append(0.0)
                        self.neg += 1
                    except:
                        continue
    def add_to_hit(self, path):
        if (self.nnet != None):
            warnings.warn("Adding data after initialization of nnet")
        with open(path) as f:
            for line in f:
                for word in utils.simple_preprocess(line):
                    try:
                        self.hitx.append(self.model.wv[word])
                        self.hity.append(1.0)
                        self.pos += 1
                    except:
                        continue
    def train_classifier(self, shuffle=True, epochs=5, batch_size=16):
        hitx = np.array(self.hitx)
        hity = np.array(self.hity)
        if (self.nnet == None):
            self.build_nnet()
        self.nnet.fit(
            x=hitx,
            y=hity,
            shuffle=shuffle,
            epochs=epochs,
            batch_size=batch_size
            )
        self.fg = self.FastGetter(self.nnet, self.model.wv)
    def filter_doc(self, path, out_path, factor = 0.5):
        with open(out_path, "w", encoding="utf-8") as out:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    li = []
                    for sentence in line.split("."):
                        accu = 0.0
                        cnt = 0
                        for word in utils.simple_preprocess(sentence):
                            try:
                                accu += self.fg.get(word)
                                cnt+=1
                            except:
                                continue
                        try:
                            accu /= cnt
                            li.append((accu, sentence))

                        except:
                            continue
                    li.sort(reverse=True)
                    li = [i for _, i in li]
                    out.write(".".join(li[:round(len(li)*factor)]) + "\n")





