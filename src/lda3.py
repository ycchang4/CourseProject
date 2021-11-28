import logging
import numpy as np
import pandas as pd
import os
from gensim.models import LdaModel

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary


docs = [] # corpus, where each string is a document

with open('textretrieval.txt', 'r') as fin:
    for line in fin:
        docs.append(line.strip())

print(len(docs))

doc_num = 2

docs = docs[doc_num:doc_num+1]

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

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=2)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

print(docs[0])

# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

# Train the model on the corpus.
lda = LdaModel(common_corpus, num_topics=10)


from gensim.test.utils import datapath

# Save model to disk.
temp_file = datapath("model")
lda.save(temp_file)

# Load a potentially pretrained model from disk.
lda = LdaModel.load(temp_file)

#Remove rare and common tokens.
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=1, no_above=1)

from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.corpus import stopwords

my_stop_words = STOPWORDS.union(set(stopwords.words('english')))

with open('stopwords.txt', 'w') as fout:
    fout.write(str(my_stop_words))

del_ids = [k for k,v in dictionary.items() if v in my_stop_words or len(v) == 1]

# remove unwanted word ids from the dictionary in place
dictionary.filter_tokens(bad_ids=del_ids)

other_corpus = [dictionary.doc2bow(doc) for doc in docs]

unseen_doc = other_corpus[0]
vector = lda[unseen_doc]  # get topic probability distribution for a document

lda.update(other_corpus)
vector = lda[unseen_doc]


lda = LdaModel(common_corpus, num_topics=50, alpha='auto', eval_every=5)  # learn asymmetric alpha from data