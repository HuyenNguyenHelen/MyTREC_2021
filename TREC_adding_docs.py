
import pandas as pd
pRUN1 = r"C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\PROJECTS\TREC_Summer2021\RESULTS\Bhanu_sent\2021\RERUN\RUNS_RawOutput\RUN1.csv"
pRUN2_2 = r"C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\PROJECTS\TREC_Summer2021\RESULTS\Bhanu_sent\2021\RERUN\RUNS_added_more_doc\RUN2_additional.csv"
pRUN2_1 =  r"C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\PROJECTS\TREC_Summer2021\RESULTS\Bhanu_sent\2021\RERUN\RUNS_RawOutput\Run2.csv"
pRUN3 = r"C:\Users\huyen\OneDrive - UNT System\A_PhD_PATH\PROJECTS\TREC_Summer2021\RESULTS\Bhanu_sent\2021\RERUN\RUNS_RawOutput\RUN3.csv"

def openFile (path):
    with open(path, 'r', encoding = 'utf-8') as f:
        data = pd.read_csv(f, delimiter = '\t', header = None)
    print(data.head())
    data.columns = ['queryID', 'docID', 'drop_score', 'score', 'unk']
    return data
run1 = openFile(pRUN1)
run2_2 = openFile(pRUN2_2)
run2_1 = openFile(pRUN2_1)
run3 = openFile(pRUN3)

#### RUN2 concat
# Drop query groups from original RUN2 that appears in additional RUN2 file
drop_groups = [query for query in run2_2['queryID'].unique() ]
for q in drop_groups:
    run2_1 = run2_1.drop(run2_1.groupby(by = ['queryID']).get_group(q).index)

# Concat RUN2-1 and RUN2-2
run2 = pd.concat([run2_1, run2_2])

#### Adding docs from RUN2 to RUN1 and RUN3 if those query groups lack doc

needDocQ_run1, needDocQ_run3 = [],[]
for i in range(1, 76):
    if len(run1[run1['queryID']==i]) <1000:
        needDocQ_run1.append(i)

for i in range(1, 76):
    if len(run3[run3['queryID']==i])<1000:
        needDocQ_run3.append(i)

def AddingDoc_toDF (run1,run2, needDocQ_run1 ):
    run1["rank"] = run1.groupby("queryID")["score"].rank("dense", ascending=False)
    added_df = []
    for q in needDocQ_run1:
        run1_docID_q = run1[run1['queryID']==q]['docID'].unique()
        # Remove duplicates from RUN2
        sub_run2 = run2[run2['queryID']==q].loc[~(run2[run2['queryID']==q]['docID'].isin(run1_docID_q))]
        print(sub_run2)
        # Geting top number of docs needed for each query
        added = sub_run2[sub_run2['queryID']==q].head(1000- len(run1[run1['queryID']==q]))
        added ['rank'] = [i+len(run1[run1['queryID']==q]) for i in range(1,len(added)+1)]
        added_df.append(added)
        #print(added_df)
    added_df_concat = pd.concat(added_df)
    new_run1 = pd.concat([run1,added_df_concat])
    new_run1 = new_run1.drop(columns = ['score'])
    new_run1 = new_run1.sort_values(by=['queryID','rank'], ascending = True).groupby('queryID').head(1000)
    new_run1['score'] = new_run1['rank'].apply(lambda x: (1000-x)/1000)
    new_run1 = new_run1[['queryID', 'docID', 'drop_score','score']]
    return new_run1
new_run1 = AddingDoc_toDF (run1,run2, needDocQ_run1 )
new_run3 = AddingDoc_toDF (run3,run2, needDocQ_run3 )

