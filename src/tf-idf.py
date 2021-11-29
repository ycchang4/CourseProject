# from typing import Dict, Iterable, List

# from allennlp.data import DatasetReader, Instance
# from allennlp.data.fields import Field, LabelField, TextField
# from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
# from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer


# # Input
# text: TextField

# # Output
# label: LabelField


# @DatasetReader.register('classification-tsv')
# class ClassificationTsvReader(DatasetReader):
#     def __init__(self):
#         self.tokenizer = SpacyTokenizer()
#         self.token_indexers = {'tokens': SingleIdTokenIndexer()}

#     def _read(self, filepath: str) -> Iterable[Instance]:
#         with open('textretrieval.txt', 'r') as lines:
#             for line in lines:
#                 text, label = line.strip().split('\t')
#                 text_field = TextField(self.tokenizer.tokenize(text),
#                                        self.token_indexers)
#                 label_field = LabelField(label)
#                 fields = {'text': text_field, 'label': label_field}
#                 yield Instance(fields)




# TF-IDF with Scikit Learn #
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from nltk.corpus import stopwords
import glob
import re


docs = [] # corpus, where each string is a document

with open('textretrieval.txt', 'r') as fin:
    for line in fin:
        docs.append(line.strip())

def remove_stops(text, stops):
    #remove all the numbers#
    pattern = r'[0-9]'
    text = re.sub(pattern, '', text)
    words = text.split()
    final = []
    for word in words:
        if word not in stops:
            final.append(word)
    final = " ".join(final)
    final = final.translate(str.maketrans("", "", string.punctuation))
    final = "".join([i for i in final if not i.isdigit()])
    while "  " in final:
        final = final.replace("  ", " ")
    return (final)



def clean_docs(docs):
    stops = stopwords.words("english")
    final = []
    for doc in docs:
        clean_doc = remove_stops(doc, stops)
        final.append(clean_doc)
    return (final)

cleaned_docs = clean_docs(docs)
print(cleaned_docs[0:1])


vectorizer = TfidfVectorizer(
                                lowercase=True,
                                max_features=100,
                                max_df=0.8,
                                min_df=5,
                                ngram_range = (1,3),
                                stop_words = "english"

                            )

vectors = vectorizer.fit_transform(cleaned_docs)

feature_names = vectorizer.get_feature_names()

dense = vectors.todense()
denselist = dense.tolist()

all_keywords = []

for docs in denselist:
    x = 0
    keywords = []
    for word in docs:
        if word > 0:
            keywords.append(feature_names[x])
        x = x + 1
    all_keywords.append(keywords)
    print(all_keywords[0])