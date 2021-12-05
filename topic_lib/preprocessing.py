### @credit: gensim tutorial on LDA and ensembleLda
import gensim
from spacy.lang.en.stop_words import STOP_WORDS
def tokenize(sentence):
    return [token for token in sentence.split() if token not in STOP_WORDS]
import re
def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    return re.sub(r'\s{2,}', ' ', sentence)
from gensim.models.phrases import Phrases, Phraser
def build_phrases(sentences):
    phrases = Phrases(sentences,
                      min_count=1,
                      threshold=10,
                      progress_per=1000)
    return Phraser(phrases)
def sentence_to_bi_grams(phrases_model, sentence):
    return ' '.join(phrases_model[sentence])

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
for handler in logger.handlers:
    handler.setLevel(logging.DEBUG)

p = []
with open("phrases.txt") as f:
    for line in f:
        phrase = line.split()
        prefix = phrase[0]
        for x in phrase[1:]:
            p.append((prefix, x))
            prefix += "_" + x
p = dict(p)

docs = []
with open("textretrieval.txt", encoding='utf-8') as f:
    for line in f:
        docs.append(line.split('.'))
with open("textanalytics.txt") as f:
    for line in f:
        docs.append(line.split('.'))
with open("noncs.dat", encoding='utf-8') as f:
    for line in f:
        docs.append(line.split('.'))
with open("cs125.dat", encoding='utf-8') as f:
    for line in f:
        docs.append(line.split('.'))
with open("bkgd.dat", encoding='utf-8') as f:
    for line in f:
        docs.append(line.split('.'))
print(len(docs))
docs = [[clean_sentence(s) for s in sentences] for sentences in docs]
docs = [[tokenize(s) for s in sentences] for sentences in docs]


phrased = []
for sentences in docs:
    ps = []
    for s in sentences:
        i = 0
        new_s = []
        while i < len(s) - 1:
            if (s[i+1] == p.get(s[i], False)):
                new_s.append(s[i] + "_" + s[i+1])
                i+=2
            else:
                new_s.append(s[i])
                i+=1
        nns =[]
        i = 0
        while i < len(new_s) - 1:
            if (new_s[i+1] == p.get(new_s[i], False)):
                nns.append(new_s[i] + "_" + new_s[i+1])
                i+=2
            else:
                nns.append(new_s[i])
                i+=1
            ps.append(nns)
    phrased.append(ps)
len(phrased)

phrased_doc = []
for doc in phrased:
    d = []
    for s in doc:
        d+=s
    phrased_doc.append(d)
# print(phrased_doc)]

from gensim.models import LdaModel
from gensim.corpora import Dictionary
import random

dictionary = Dictionary(phrased_doc)
dictionary.filter_extremes(no_below=2, no_above=0.1)

corpus = [dictionary.doc2bow(doc) for doc in phrased_doc]
num_topics = 86*2
chunksize = 160 # how many documents to process at a time
passes = 20 # epochs
iterations = 100
eval_every = 10

# Make a index to word dictionary.
temp = dictionary[0] 
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

### after word2vec skimming
reduced = []
with open("reducedretrieval.dat", encoding='utf-8') as f:
    for line in f:
        reduced.append(line.split('.'))
with open("reducedanalytics.dat", encoding='utf-8') as f:
    for line in f:
        reduced.append(line.split('.'))
reduced = [[clean_sentence(s) for s in sentences] for sentences in reduced]
reduced = [[tokenize(s) for s in sentences] for sentences in reduced]

phrased = []
for sentences in reduced:
    ps = []
    for s in sentences:
        i = 0
        new_s = []
        while i < len(s) - 1:
            if (s[i+1] == p.get(s[i], False)):
                new_s.append(s[i] + "_" + s[i+1])
                i+=2
            else:
                new_s.append(s[i])
                i+=1
        nns =[]
        i = 0
        while i < len(new_s) - 1:
            if (new_s[i+1] == p.get(new_s[i], False)):
                nns.append(new_s[i] + "_" + new_s[i+1])
                i+=2
            else:
                nns.append(new_s[i])
                i+=1
        ps.append(nns)
    phrased.append(ps)
    
phrased_doc = []
for doc in phrased:
    d = []
    for s in doc:
        d+=s
    phrased_doc.append(d)
reduced_corpus = [dictionary.doc2bow(doc) for doc in phrased_doc]
# phrased_doc

def takelast(x):
    return x[-1]

with open("results.dat", "w") as out:
    print(len(phrased_doc))
    for k in range(len(reduced)):
        top_topics = model.get_document_topics(reduced_corpus[k]) # [(topic_id, prob)]
        ####
        top_topics.sort(key=takelast, reverse=True)
        for topic in top_topics:
            print('topic {}'.format(k))
            topic_term_distribution = model.get_topic_terms(topic[0])
            for term in topic_term_distribution:
                print('{}, {}'.format(dictionary[term[0]], term[1]))
                out.write(dictionary[term[0]] + ",")
            print('prob: {}'.format(topic[-1]))
        out.write("\n")




from gensim.test.utils import datapath
# Save model to disk.
temp_file = datapath("model2.lda")
model.save(temp_file)