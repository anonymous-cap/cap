import json
import os
import random
import pandas as pd
from fuzzywuzzy import fuzz

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
        return data


data_ori_dev=load_json("ori_dev.json")
data_ori_val=load_json("ori_val.json")

similarity_threshold = 30


def check_similarity(question, item_input, threshold):

    question_similarity = fuzz.ratio(question.strip(), item_input)

    return question_similarity >= threshold


if __name__ == "__main__":
    for d in ['dev', 'val','test']:
        for filename in sorted(os.listdir(d)):
            if filename.endswith('.csv'):
                path = os.path.join(d, filename)

                df = pd.read_csv(path)
                for i in range(len(df)):
                    question = df.iloc[i]["question"]
                    opt_a = df.iloc[i]["A"]
                    opt_b = df.iloc[i]["B"]
                    opt_c = df.iloc[i]["C"]
                    opt_d = df.iloc[i]["D"]


                    for item in data_ori_dev:
                        if check_similarity(question, item["input"], similarity_threshold):
                            print("------------------------------------")
                            print(path)
                            print("data_ori_dev")
                            print(question, f"A: {opt_a} B: {opt_b} C:{opt_c} D:{opt_d}")
                            print(item["input"])
                            print("*********")

                    for item in data_ori_val:
                        if check_similarity(question, item["input"], similarity_threshold):
                            print("------------------------------------")
                            print(path)
                            print("data_ori_val")
                            print(question, f"A: {opt_a} B: {opt_b} C:{opt_c} D:{opt_d}")
                            print(item["input"])
                            print("*********")