import pandas as pd
import os
import xml.etree.ElementTree as ET

#### Reformat PRF term extraction with Metamap######
def reformat(path):
    expanded_v1 = open(path, "r")
    data = expanded_v1.readlines()[:-1]
    docID, query_concept, extended_concept, semantic_type, score, CUI = [], [], [], [], [], []
    for line in data:
        docID.append(line.split('|')[0])
        query_concept.append(line.split('|')[6].split('"')[-2])
        extended_concept.append(line.split('|')[3])
        score.append(line.split('|')[2])
        CUI.append(line.split('|')[4])
        semantic_type.append(line.split('|')[5])
        df = pd.DataFrame(zip(docID, query_concept, semantic_type, extended_concept, score, CUI),
                          columns=['TopicID', 'query_concept', 'semantic_type', 'extended_concept', 'score', 'CUI'])
    return df


des_outputDF = reformat(r"/home/junhua/Downloads/huyen/PRF_terms_top25_2019")
des_outputDF['TopicID'] = pd.to_numeric(des_outputDF['TopicID'])
topics = des_outputDF.groupby('TopicID')
dic1 = {}
expanded_query = []
for i in range(1, 41):
    if i != 31:
        expanded_query.append(list(topics.get_group(i)['extended_concept']))
        dic1[i] = list(topics.get_group(i)['extended_concept'])
        #print(list(topics.get_group(i)['extended_concept']))
    else:
        pass

########## Extracting original terms in the query ###########
tree = ET.parse(r"/home/junhua/Downloads/huyen/topics2019.xml")
root = tree.getroot()

i = 1
while i < len(root):
    try:
        for j in range(len(root[i]) - 1):
            # print('.............', root[i][j].text)
            dic1[i].append(root[i][j].text)
    except:
        pass
    i += 1

############## ClinicalBERT similarity ################
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

#Initialize our model and tokenizer:
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


if os.path.exists(r'/home/junhua/Downloads/huyen/temp1.txt'):
    os.remove(r'/home/junhua/Downloads/huyen/temp1.txt')
else:
    f = open(r'/home/junhua/Downloads/huyen/temp1.txt','w')
    x = list(range(1,41))
    x.remove(31)
    for i in x:
        # initialize dictionary: stores tokenized sentences
        tokens = {'input_ids': [], 'attention_mask': []}
        for query in dic1[i]:
            new_tokens = tokenizer.encode_plus(query, max_length=128,
                                               truncation=True, padding='max_length',
                                               return_tensors='pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])
        # reformat list of tensors to single tensor
        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
        # Process tokens through model:
        outputs = model(**tokens)
        # The dense vector representations of text are contained within the outputs 'last_hidden_state' tensor
        embeddings = outputs[0] #outputs.last_hidden_state
        # To perform this operation, we first resize our attention_mask tensor:
        attention_mask = tokens['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        # Then we sum the remained of the embeddings along axis 1:
        summed = torch.sum(masked_embeddings, 1)
        # Then sum the number of values that must be given attention in each position of the tensor:
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        # Let's calculate cosine similarity for sentence 0:
        # convert from PyTorch tensor to numpy array
        mean_pooled = mean_pooled.detach().numpy()
        # calculate
        similarity = cosine_similarity([mean_pooled[0]],mean_pooled[1:])
        a = zip(dic1[i][1:], similarity.reshape(-1))
        for j in a:
            print('topic{}'.format(i), ',', dic1[i][0], ',', j[0], ',', j[1], file=f)
    f.close()