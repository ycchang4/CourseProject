from rake_nltk import Rake

r = Rake()
with open('textanalytics.txt', 'r') as fin:
    i = 0
    for line in fin:
        i += 1
        print('lecture {}'.format(i))
        s = line.strip()
        r.extract_keywords_from_text(s)
        print(r.get_ranked_phrases()[0:10])
