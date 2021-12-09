# credit: https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-download-auto-examples-tutorials-run-lda-py
# ldamodel: https://radimrehurek.com/gensim/models/ldamodel.html

import logging
from os import write
import numpy as np
import sys

def write_to_file(f, lst):
    with open(f, 'w') as fout:
        fout.write(str(lst))

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

docs = [] # corpus, where each string is a document

in_f = 'textretrieval.txt'
out_f = 'lda_results.txt'
argc = len(sys.argv)

if argc > 1:
    in_f = sys.argv[1]

if argc > 2:
    out_f = sys.argv[2]

with open(in_f, 'r') as fin:
    for line in fin:
        docs.append(line.strip())

#print(len(docs))

doc_idx = 0
doc_num = len(docs)

#docs = docs[doc_idx:doc_idx+doc_num]

from nltk.tokenize import RegexpTokenizer

# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

# Remove words that are only one character.
docs = [[token for token in doc if len(token) > 1] for doc in docs]

# Lemmatize the documents.
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

# Compute bigrams.
from gensim.models import Phrases

bigram = Phrases(docs, min_count=2)
trigram = Phrases(bigram[docs], min_count=2) # 4-grams
write_to_file('trigrams.txt', trigram[bigram[docs[0]]])

docs_without_unigrams = []

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
for idx in range(len(docs)):
    docs_without_unigrams.append([])
    for token in trigram[bigram[docs[idx]]]:
        if '_' in token or token == 'bm25':
            # Token is a bigram, add to document.
            docs[idx].append(token)
            docs_without_unigrams[-1].append(token)

# print(docs[0])
# print(docs_without_unigrams)

docs = docs_without_unigrams

# Remove rare and common tokens.
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=2, no_above=0.2)

from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.corpus import stopwords

my_stop_words = STOPWORDS.union(set(stopwords.words('english')))

with open('stopwords.txt', 'w') as fout:
    fout.write(str(my_stop_words))

del_ids = [k for k,v in dictionary.items() if v in my_stop_words or len(v) == 1]

# remove unwanted word ids from the dictionary in place
dictionary.filter_tokens(bad_ids=del_ids)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]

# print('Number of unique tokens: %d' % len(dictionary))
# print('Number of documents: %d' % len(corpus))

# Train LDA model.
from gensim.models import LdaModel

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

# Set training parameters.
num_topics = 5 * len(corpus)
chunksize = len(corpus) # how many documents to process at a time
passes = 100 # epochs
iterations = 400
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=1
)

with open(out_f, 'w') as fout:
    for lecture in range(0, len(corpus)):
        print('working on lecture {}'.format(lecture+1))
        fout.write('lecture {}\n'.format(lecture+1))
        # top_topics = model.top_topics(corpus) #, num_words=20)
        top_topics = model.get_document_topics(corpus[lecture]) # [(topic_id, prob)]

        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        # avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        # print('Average topic coherence: %.4f.' % avg_topic_coherence)

        # from pprint import pprint
        # pprint(top_topics)

        i = 1
        for topic in top_topics:
            fout.write('topic {}\n'.format(i))
            i+=1
            topic_term_distribution = model.get_topic_terms(topic[0], topn=40)
            for term in topic_term_distribution:
                fout.write('{}, {}\n'.format(dictionary[term[0]], term[1]))
            fout.write('prob: {}\n'.format(topic[-1]))

        # with open('tokens.txt', 'w') as fout:
        #     fout.write(str(dictionary.token2id))
