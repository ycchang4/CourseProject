results = []

i = -1
with open('lda_results.txt', 'r') as f:
    for line in f:
        if line.strip().split(' ')[0] == 'lecture':
            i += 1 
            results.append(set()) 
        elif line.strip().split(' ')[0] in ['topic', 'prob:'] :
            continue
        else:
            results[-1].add(line.strip().split(',')[0])

#print(results)

from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

threshold = 0.4

print('lecture, #concepts retrieved, recall')

i = -1
with open('annotated_set.csv', 'r') as f:
    for line in f:
        i += 1
        concepts = line.strip().split(';')
        n = 0
        found = 0
        empty_concepts = 0
        for concept in concepts:
            # calculate recall
            n += 1
            words = concept.split(' ')
            words = [lemmatizer.lemmatize(token).lower() for token in words]
            #print(words)
            for res in results[i]:
                res_words = res.split('_')
                j = 0
                k = 0
                for word in words:
                    if word == '':
                        continue
                    j += 1
                    if word in res_words:
                        k += 1
                if j == 0:
                    empty_concepts += 1
                    break
                if k/j >= threshold:
                    found += 1
                    break
        print('lecture {}: {}, {}'.format(i, found, found/(len(concepts)-empty_concepts)))

