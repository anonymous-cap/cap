import json
import os
import pandas as pd
import pdb
from tqdm import tqdm

def load_json(file):
    with open(file, "r") as f:
        data=json.load(f)
    return data

def jaccard_similarity(string1, string2):
    set1, set2 = set(string1), set(string2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)


    if len(union) == 0:
        return 0

    return len(intersection) / len(union)

fineval_val_data=load_json("fineval_val.json")

for dire in [ "test"]:
    dfs = {}
    with open(f'output_{dire}.txt', 'w') as f:
        for filename in tqdm(os.listdir(dire)):
            if filename.endswith('.csv'):
                file_path = os.path.join(dire, filename)

                df = pd.read_csv(file_path)

                dfs[filename] = df

                for i in range(len(df)):
                    if df.iloc[i]['question']:
                        ceval_string=df.iloc[i]['question']+f"['A':{ df.iloc[i]['A']}, 'B': {df.iloc[i]['B']}, 'C':{df.iloc[i]['C']}, 'D':{df.iloc[i]['D']}]"
                        for j in range(len(fineval_val_data)):
                            fineval_string= fineval_val_data[j]['input'].replace('请回答选择题，从选项A，B，C，D中选择，回答问题。\n问题: ', '').replace('选项：','').replace('答案： ','')
                            dis=jaccard_similarity(ceval_string, fineval_string)
                            if dis>0.7:
                                f.write("CEVAL " + dire + "\n")
                                f.write(ceval_string + "\n")
                                f.write("FinEval Val\n")
                                f.write(fineval_string + "\n")

