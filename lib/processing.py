### @credit: gensim tutorial on LDA and ensembleLda
from spacy.lang.en.stop_words import STOP_WORDS
import re
from gensim.models.phrases import Phrases
from gensim.models import LdaModel
from gensim.corpora import Dictionary

class ConceptExtract:
    def __init__(self, phrase_path):
        p = []
        with open(phrase_path) as f:
            for line in f:
                phrase = line.split()
                prefix = phrase[0]
                for x in phrase[1:]:
                    p.append((prefix, x))
                    prefix += "_" + x
        self.p = dict(p)
        self.docs = []
        self.phrased_doc = []
        self.model = None
        self.dictionary = None
        self.reduced = []

    def __tokenize(self,sentence):
            return [token for token in sentence.split() if token not in STOP_WORDS]

    def __clean_sentence(self,sentence):
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
        return re.sub(r'\s{2,}', ' ', sentence)
    
    @staticmethod
    def takelast(x):
        return x[-1]

    def add_corpus(self, path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                self.docs.append(line.split('.'))
    
    def preprocess(self, lb = 2, ub = 0.5 ):
        docs = [[self.__clean_sentence(s) for s in sentences] for sentences in self.docs]
        docs = [[self.__tokenize(s) for s in sentences] for sentences in docs]
        phrased = []
        for sentences in docs:
            ps = []
            for s in sentences:
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
                ps.append(nns)
            phrased.append(ps)
        for doc in phrased:
            d = []
            for s in doc:
                d+=s
            self.phrased_doc.append(d)

    def createLDA(self, num_topics, chunksize, lb = 2, ub=0.1, passes=20, iterations=100,eval_every=None):
        self.dictionary = Dictionary(self.phrased_doc)
        self.dictionary.filter_extremes(no_below=lb, no_above=ub)
        corpus = [self.dictionary.doc2bow(doc) for doc in self.phrased_doc]
        # Make a index to word dictionary.
        temp = self.dictionary[0]
        id2word = self.dictionary.id2token

        self.model = LdaModel(
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
    def add_target(self, path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                self.reduced.append(line.split('.'))

    def extract_concept(self, path):
        reduced = [[self.__clean_sentence(s) for s in sentences] for sentences in self.reduced]
        reduced = [[self.__tokenize(s) for s in sentences] for sentences in reduced]

        phrased = []
        for sentences in reduced:
            ps = []
            for s in sentences:
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
                ps.append(nns)
            phrased.append(ps)
            
        phrased_doc = []
        for doc in phrased:
            d = []
            for s in doc:
                d+=s
            phrased_doc.append(d)
        reduced_corpus = [self.dictionary.doc2bow(doc) for doc in phrased_doc]

        with open(path, "w") as out:
            print(len(phrased_doc))
            for k in range(len(reduced)):
                top_topics = self.model.get_document_topics(reduced_corpus[k]) # [(topic_id, prob)]
                ####
                top_topics.sort(key=self.takelast, reverse=True)
                for topic in top_topics:
                    print('topic {}'.format(k))
                    topic_term_distribution = self.model.get_topic_terms(topic[0])
                    for term in topic_term_distribution:
                        print('{}, {}'.format(self.dictionary[term[0]], term[1]))
                        out.write(self.dictionary[term[0]] + ",")
                    print('prob: {}'.format(topic[-1]))
                out.write("\n")

