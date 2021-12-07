from gensim.utils import chunkize
import processing
import phrasing
import wordvec
import gensim

text8 = gensim.models.word2vec.Text8Corpus("text8")


def omakase():
    phrase_path = "phrases.txt"
    x = phrasing.ConceptPhraser()
    x.add_background("bkgd.txt")
    x.add_background("cs125.dat")
    x.add_background("noncs.dat")
    x.add_target("textanalytics.txt")
    x.add_target("textretrieval.txt")
    x.process()
    x.save(phrase_path)
    #################################
    y = wordvec.WordVecFilter()
    lee = y.add_phrased_corpus(None)
    
    bk = y.add_phrased_corpus("bkgd.txt")
    cs = y.add_phrased_corpus("cs125.dat")
    non = y.add_phrased_corpus("noncs.dat")
    tr = y.add_phrased_corpus("textanalytics.txt")
    ta = y.add_phrased_corpus("textretrieval.txt")
    id8 = y.add_basic_corpus("text8")
    y.train_corpus(lee)
    y.train_corpus(bk)
    y.train_corpus(cs)
    y.train_corpus(non)
    y.train_corpus(tr)
    y.train_corpus(ta)
    y.train_corpus(id8)
    y.add_to_hit("hits.dat")
    y.add_to_miss("miss.dat")
    y.build_nnet()
    y.train_classifier()
    y.filter_doc("textanalytics.txt","reducedanalytics.dat")
    y.filter_doc("textretrieval.txt", "reducedretrieval.dat")
    ##################################
    z = processing.ConceptExtract(phrase_path)
    bk = z.add_corpus("bkgd.dat")
    cs = z.add_corpus("cs125.dat")
    non = z.add_corpus("noncs.dat")
    tr = z.add_corpus("textanalytics.txt")
    ta = z.add_corpus("textretrieval.txt")
    z.preprocess()
    num_topics = 200
    chunkize = 160
    z.createLDA(num_topics, chunkize)
    z.add_target("reducedretrieval.dat")
    z.add_target("reducedanalytics.dat")
    out = "results.out"
    z.extract_concept(out)



omakase()