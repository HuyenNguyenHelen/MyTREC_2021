import pandas as pd
from numpy import arange
import os
import os.path
import glob
import xml.etree.ElementTree as ET

path  = r"C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\TREC_Summer2021\RESULTS\2019\BoolQuery_scores_2019_best.csv"
with open(path, 'r', encoding = 'utf-8') as file:
    df = pd.read_csv(file, delimiter = '\t')
df.columns = ['topicID', 'docID', 'score']
topics = df.groupby('topicID')

files = {}
topic= arange (1, 41, 1)
#final_dict = {}
for i in topic:
    try:
    #print((topics.get_group(i).head(20)))
        files[i]=list(topics.get_group(i).head(25)['docID']) 
    except:
        print(i)


import io

with io.open(
        r"C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\TREC_Summer2021\METHOD\Query processing\Query_expansion\PRF\PRF-docs_top25_2019.txt",
        "w", encoding="utf-8") as f:
    count = 1
    final_result = {}
    for key, value in files.items():
        brief_titles = []
        for fname in value:
            xml_files = glob.glob(
                r'C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\TREC_Summer2021\data\TREC-2019\clinical_trials*\\*\\' + fname + ".xml",
                recursive=True)
            # print(xml_files)
               tree = ET.parse(xml_files[0])
               root = tree.getroot()
               result = {}
               brief_titles.append(root[2].text)
               titles = ','.join(brief_titles)
               result[key] = titles
               print('{} | {}'.format(key, titles), file=f)
               final_result.update(result)
      
for key, value in final_result.items():
    print(key, value)

df = pd.DataFrame.from_dict(final_result, orient='index')
with open (r'C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\TREC_Summer2021\RESULTS\\PRF-docs_top25_2019.txt', 'w', encoding = 'utf-8', newline = '') as f:
    df.to_csv(f)
