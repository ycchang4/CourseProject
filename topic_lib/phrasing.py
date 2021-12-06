
from spacy.lang.en.stop_words import STOP_WORDS
import re
from gensim.models.phrases import Phrases, Phraser

class ConceptPhraser:
    def __tokenize(self,sentence):
        return [token if token not in STOP_WORDS else "@" for token in sentence.split() ]

    def __clean_sentence(self,sentence):
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
        return re.sub(r'\s{2,}', ' ', sentence)
    
    def __build_phrases(self,sentences):
        bi = Phrases(sentences,
                        min_count=3,
                        progress_per=1000)
        tri = Phrases(bi[sentences], min_count=3)
        return bi, tri
    def __sentence_to_bi_grams(self,big,trig, sentence):
        return ' '.join(trig[big[sentence]])

    tar = []
    bg = []
    phrases = []
    selected = []
    def add_target(self, path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                self.tar+=line.split('.')
    def add_background(self, path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                self.bg+=line.split('.')
    def __build(self, x):
        
        sentences = [self.__clean_sentence(s) for s in x]
        sentences = [self.__tokenize(s) for s in sentences]
        bi, tri = self.__build_phrases(sentences)
        phrased = [self.__sentence_to_bi_grams(bi,tri, s) for s in sentences]
        
        ph = [w for s in phrased for w in s.split() if "_" in w]
        phc = [(ph.count(x), x) for x in set(ph)]
        return phc

    def process(self, tol=2.99):
        phc = self.__build(self.tar)
        bphc = self.__build(self.bg)
        
        back_phc = dict([(b,a) for a,b in bphc])
        phc = [(a / back_phc.get(x,1), x) for a,x in set(phc)]
        self.selected = [b for a,b in phc if a > tol ]
    def save(self, path):
        phrases= [x.replace("_", " ") for x in self.selected]   
        with open(path, "w") as out:
            for x in phrases:
                out.write(x + "\n")
    




