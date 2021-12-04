from os import write
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.corpus import stopwords
import gensim.models

def write_to_file(f, lst):
    with open(f, 'w') as fout:
        fout.write(str(lst))

def read_docs(f):
    docs = [] # corpus, where each string is a document

    with open(f, 'r') as fin:
        for line in fin:
            docs.append(line.strip())

    return docs

def clean_docs(docs):
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
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    return docs

def get_ngrams(docs):
    # Compute bigrams.
    bigram = Phrases(docs, min_count=1)
    trigram = Phrases(bigram[docs], min_count=1) # 4-grams
    write_to_file('trigrams.txt', trigram[bigram[docs[0]]])

    # Add bigrams and trigrams to docs
    for idx in range(len(docs)):
        for token in trigram[bigram[docs[idx]]]:
            if '_' in token:
                docs[idx].append(token)

    return docs

def get_vocab(docs):
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=1, no_above=1)
    my_stop_words = STOPWORDS.union(set(stopwords.words('english')))

    write_to_file('stopwords.txt', my_stop_words)

    del_ids = [k for k,v in dictionary.items() if v in my_stop_words or len(v) == 1]

    # remove unwanted word ids from the dictionary in place
    dictionary.filter_tokens(bad_ids=del_ids)

    write_to_file('vocab.txt', dictionary.token2id)

    return dictionary

def get_tf_idfs(vocab, corpus):
    tf = []
    idf = defaultdict(lambda: 0)
    for doc in corpus:
        tf.append(defaultdict(lambda: 0))
    for token in vocab:
        i = 0
        for doc in corpus:
            doc = dict(doc)
            if token in doc.keys():
                idf[token] += 1
                tf[i][token] = doc[token]
            i += 1

    write_to_file('idf.txt', idf)

    return tf, idf

def get_wordvec():
    text8 = gensim.models.word2vec.Text8Corpus("text8")
    model = gensim.models.Word2Vec(sentences=text8)
    return model

if __name__ == '__main__':
    docs = read_docs('textretrieval.txt')
    docs = clean_docs(docs)
    docs = get_ngrams(docs)
    vocab = get_vocab(docs)
    corpus = [vocab.doc2bow(doc) for doc in docs]
    tf, idf = get_tf_idfs(vocab, corpus)
    model = get_wordvec()
    
