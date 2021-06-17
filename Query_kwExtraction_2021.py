
# Loading the Query [2016/2021]
## Query 2016
import pandas as pd
import xml.etree.ElementTree as ET

tree = ET.parse(r"C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\TREC_Summer2021\data\topics2016.xml")
root = tree.getroot()

queryID, note, description, summary=[],[],[], []

 
i=0
while i<len(root):
    queryID.append(i+1)
    for j in range (len(root[i])):
        if root[i][j].tag == 'note':
            note.append(root[i][j].text.replace('\n',''))
        if root[i][j].tag == 'summary':
            summary.append(root[i][j].text)
        if root[i][j].tag == 'description':
            description.append(root[i][j].text)
  
    i+=1
    
df = pd.DataFrame(zip(queryID, note, description, summary), columns =['queryID', 'note', 'description', 'summary'] )

# Applying Keyword Extraction 
#!pip install git+https://github.com/LIAAD/yake
import yake
       
# Building Yake model
def fitYAKE (text):
    # Specify parameters
    language = "en"
    max_ngram_size = 3
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords =30

    # Yake model with specified parameters
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size,  dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    return keywords

# rank the output by score
def rankOutput (list_tuples):
    to_dict = dict(list_tuples)
    rank_dict = {k:v for k, v in sorted(to_dict.items(), key=lambda item: item[1], reverse = True)}
    return ','.join([k for k in rank_dict.keys()])

sum_kw = df[['summary']].applymap(fitYAKE)
note_kw = df[['note']].applymap(fitYAKE)
des_kw = df[['description']].applymap(fitYAKE)

sum_kw = sum_kw.applymap(rankOutput)
note_kw = note_kw.applymap(rankOutput)
des_kw = des_kw.applymap(rankOutput) 

df['summary_keyword'] = sum_kw
df['note_keyword'] = note_kw
df['description_keyword'] = des_kw
df = df[['queryID', 'summary','summary_keyword','description', 'description_keyword', 'note', 'note_keyword']]

with open (r"C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\TREC_Summer2021\RESULTS\Query processing\Raw_Query_Processing\2016\Keyword_Extr\KeywordExtr_Query2016_30w.csv", 'w', encoding = 'utf-8', newline = '') as file:
    df.to_csv(file)
df.head()

df.head()
