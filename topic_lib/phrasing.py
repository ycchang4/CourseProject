import gensim
from spacy.lang.en.stop_words import STOP_WORDS
def tokenize(sentence):
    return [token if token not in STOP_WORDS else "@" for token in sentence.split() ]
import re
def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    return re.sub(r'\s{2,}', ' ', sentence)
from gensim.models.phrases import Phrases, Phraser
def build_phrases(sentences):
    bi = Phrases(sentences,
                      min_count=3,
                      progress_per=1000)
    tri = Phrases(bi[sentences], min_count=3)
    return bi, tri
def sentence_to_bi_grams(big,trig, sentence):
    return ' '.join(trig[big[sentence]])


##moved 
sentences = []
with open("cs125.dat", encoding='utf-8') as f:
    for line in f:
        sentences+=line.split('.')
with open("noncs.dat", encoding='utf-8') as f:
    for line in f:
        sentences+=line.split('.')
with open("bkgd.txt", encoding='utf-8') as f:
    for line in f:
        sentences+=line.split('.')
bi, tri = build_phrases(sentences)
sentences = [clean_sentence(s) for s in sentences]
sentences = [tokenize(s) for s in sentences]
phrase_model = build_phrases(sentences)
phrased = [sentence_to_bi_grams(bi,tri, s) for s in sentences]
back_ph = [w for s in phrased for w in s.split() if "_" in w]
back_phc = [(back_ph.count(x), x) for x in set(back_ph)]
back_phc.sort()
print(back_phc)
back_phc = dict([(b,a) for a,b in back_phc])



## moved 2
sentences = []
with open("noncs.dat", encoding='utf-8') as f:
    for line in f:
        sentences+=line.split('.')
sentences = [clean_sentence(s) for s in sentences]
sentences = [tokenize(s) for s in sentences]
phrase_model = build_phrases(sentences)
phrased = [sentence_to_bi_grams(bi,tri, s) for s in sentences]
ph = [w for s in phrased for w in s.split() if "_" in w]
phc = [(ph.count(x), x) for x in set(ph)]
phc.sort()
aphc = dict([(b,a) for a,b in phc])

## moved 3
sentences = []
with open("bkgd.txt", encoding='utf-8') as f:
    for line in f:
        sentences+=line.split('.')
sentences = [clean_sentence(s) for s in sentences]
sentences = [tokenize(s) for s in sentences]
phrase_model = build_phrases(sentences)
phrased = [sentence_to_bi_grams(bi,tri, s) for s in sentences]
ph = [w for s in phrased for w in s.split() if "_" in w]
phc = [(ph.count(x), x) for x in set(ph)]
phc.sort()
bphc = dict([(b,a) for a,b in phc])

## moved 4
sentences = []
with open("cs125.dat", encoding='utf-8') as f:
    for line in f:
        sentences+=line.split('.')
sentences = [clean_sentence(s) for s in sentences]
sentences = [tokenize(s) for s in sentences]
phrase_model = build_phrases(sentences)
phrased = [sentence_to_bi_grams(bi,tri, s) for s in sentences]
ph = [w for s in phrased for w in s.split() if "_" in w]
phc = [(ph.count(x), x) for x in set(ph)]
phc.sort()
cphc = dict([(b,a) for a,b in phc])

sentences = []
with open("textanalytics.txt", encoding='utf-8') as f:
    for line in f:
        sentences+=line.split('.')
with open("textretrieval.txt", encoding='utf-8') as f:
    for line in f:
        sentences+=line.split('.')
sentences = [clean_sentence(s) for s in sentences]
sentences = [tokenize(s) for s in sentences]
bi, tri = build_phrases(sentences)
phrased = [sentence_to_bi_grams(bi, tri, s) for s in sentences]
ph = [w for s in phrased for w in s.split() if "_" in w]
phc = [(ph.count(x) / back_phc.get(x,1), x) for x in set(ph)]
phc = [(a / aphc.get(b,1), b) for a,b in phc]
phc = [(a / bphc.get(b,1), b) for a,b in phc]
phc = [(a / cphc.get(b,1), b) for a,b in phc]
phc.sort()
selected = [b for a,b in phc if a > 2.99 ]


len(selected)


phrases= [x.replace("_", " ") for x in selected]   
with open("phrases.txt", "w") as out:
    for x in phrases:
        out.write(x + "\n")








