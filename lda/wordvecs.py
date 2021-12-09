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
import numpy as np
from scipy import spatial

def write_to_file(f, lst):
    with open(f, 'w') as fout:
        fout.write(str(lst))
        fout.write('\n')

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

def get_wordvec(train):
    if train == True:
        text8 = gensim.models.word2vec.Text8Corpus("text8")
        model = gensim.models.Word2Vec(sentences=text8)
        model.save('word2vec.model')
        return model
    else:
        model = gensim.models.Word2Vec.load('word2vec.model')
        return model

if __name__ == '__main__':
    model = get_wordvec(train = True)
    docs = read_docs('textretrieval.txt')
    docs = clean_docs(docs)
    docs = get_ngrams(docs)
    vocab = get_vocab(docs)
    corpus = [vocab.doc2bow(doc) for doc in docs]

    with open('wordvecs_result.txt', 'w') as fout:
        pass
    for j in range(0, len(corpus)):
        with open('wordvecs_result.txt', 'a') as fout:
            fout.write('lecture ')
            fout.write(str(j+1))
            fout.write('\n')
        tf, idf = get_tf_idfs(vocab, corpus[0:j+1])
        ngrams = []
        for wd in dict(corpus[j]).keys():
            word = vocab[wd]
            if '_' in word and word not in ngrams:
                ngrams.append(word)
        vecs = dict()
        for word in ngrams:
            words = word.split('_')
            n = 0
            vec = [0.0 for x in range(100)]
            for wd in words:
                if wd in vocab.token2id.keys() and tf[0][vocab.token2id[wd]] > 0 and wd in model.wv:
                    k = 200
                    new_vec = model.wv[wd] /np.linalg.norm(model.wv[wd], 2) * (k+1)*tf[0][vocab.token2id[wd]] / (tf[0][vocab.token2id[wd]]+k) * (np.log(1+len(corpus)) - np.log(idf[vocab.token2id[wd]]))
                    vec += new_vec
                    n += 1
            if n > 0:
                vec /= np.linalg.norm(vec, 2)
            vecs[word] = list(vec)

        scores = dict()
        for word in vecs.keys():
            score = 0
            for word2 in vecs.keys():
                if word != word2 and not all(v == 0 for v in vecs[word]) and not all(v == 0 for v in vecs[word2]):
                    score += (1 - spatial.distance.cosine(vecs[word], vecs[word2])) #* np.log(tf[0][vocab.token2id[word2]]+1) * (np.log(len(corpus)) - np.log(idf[vocab.token2id[word2]])+1)
            scores[word] = score #* np.log(tf[0][vocab.token2id[word]]+1) * (1+np.log(len(corpus)) - np.log(idf[vocab.token2id[word]])) #* ngrams.count(word)
        
        #print(scores)
        with open('wordvecs_result.txt', 'a') as fout:
            fout.write(str(sorted(scores.items(), key =
                lambda kv:(kv[1], kv[0]))))
            fout.write('\n')

        # vv = model.wv['vector'] /np.linalg.norm(model.wv['vector'], 2)
        # print(np.sqrt(vv.dot(vv)))

        # with open('tf-idf.txt', 'w') as fout:
        #     for wd in dict(corpus[0]).keys():
        #         if wd in vocab.keys() and tf[0][wd] > 0:
        #             fout.write(str(vocab[wd]))
        #             fout.write(str(np.log(tf[0][wd]+1) * (np.log(len(corpus)) - np.log(idf[wd])+1)))
        #             fout.write('\n')
