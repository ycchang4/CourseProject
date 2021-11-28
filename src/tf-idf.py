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

def remove_stop(text, stops):
    #remove all the numbers#
    pattern = r'[0-9]'
    text = re.sub(pattern, '', text)
    

def clean_docs():
    stops = stopwords.words("english")

