# Loading the Query [2016/2021]
## Query 2021
import pandas as pd
import xml.etree.ElementTree as ET

tree = ET.parse(r"C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\TREC_Summer2021\data\topics2021.xml")
root = tree.getroot()

queryID, query_txt = [], []

i = 0
while i < len(root):
    queryID.append(i + 1)
    # for j in range (len(root[i])):
    print(root[i].text)
    query_txt.append(root[i].text)
    i += 1

df = pd.DataFrame(zip(queryID, query_txt), columns=['queryID', 'query_txt'])

# Applying Keyword Extraction
# pip install git+https://github.com/LIAAD/yake
import yake


# Building Yake model
def fitYAKE(text):
    # Specify parameters
    language = "en"
    max_ngram_size = 3
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 20

    # Yake model with specified parameters
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                                dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                                features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    return keywords

# rank the output by score
def rankOutput(list_tuples):
    to_dict = dict(list_tuples)
    rank_dict = {k: v for k, v in sorted(to_dict.items(), key=lambda item: item[1], reverse=True)}
    return rank_dict

# fit Yake
kw = df[['query_txt']].applymap(fitYAKE)
# rank the output by score
kw = kw.applymap(rankOutput)
df['keyword'] = kw
df = df[['queryID', 'query_txt', 'keyword']]

# write to file
with open (r'C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\TREC_Summer2021\METHOD\Query processing\Query_expansion\Query_kwExtraction_2021_1-3gram.csv', 'w', encoding = 'utf-8', newline = '') as file:
    df.to_csv(file)

df.head()
