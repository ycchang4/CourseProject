if __name__ == '__main__':
    with open('textanalytics.txt', 'w') as fout:
        for i in range(1, 54):
            with open('transcripts/textanalytics/tm-lec'+str(i)+'-transcription-english.vtt', 'r') as fin:
                j = 0
                for line in fin:
                    j+=1
                    if j<=5:
                        continue
                    if line.strip() == '':
                        continue
                    if (len(line)>=3) and (line[0]<='9' and line[0]>='0') and (line[1]<='9' and line[1]>='0') and line[2]==':':
                        continue
                    fout.write(line.strip()+' ')
            fout.write('\n')
