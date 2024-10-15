from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch
import json
import pdb
import os
import pandas as pd
import numpy as np
import re
def load_json(file):
    with open(file, 'r') as f:
        data=json.load(f)
    return data

def clean_text(text):

    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)


    text = re.sub(r'[\(\),.]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()


    text = re.sub(r'(\w)\1{2,}', r'\1', text)


    text = re.sub(r'\.+', '', text)

    return text

def cal_above_thresh(sim, thresh):
    above_threshold = sim >= thresh
    proportion = above_threshold.sum().item() / len(sim)
    return proportion


emb_model = SentenceTransformer("all-MiniLM-L6-v2")

sim_thre_list=[]

for type in ["train","val","test"]:
    for model in ["llama", "mistral", "finma_full","finma_nlp","baichuan","disc_fin","gpt"]:
        pred_dir = f"../{model}_pred"
        if model in ["llama", "baichuan"]:
            ori_file = os.path.join(pred_dir, f"ori_{type}_samples.json")
            mod_file = os.path.join(pred_dir, f"mod_{type}_samples.json")
        else:
            ori_file = os.path.join(pred_dir, f"ori_{type}.json")
            mod_file = os.path.join(pred_dir, f"mod_{type}.json")
        ori_data = load_json(ori_file)
        mod_data = load_json(mod_file)
        assert len(ori_data) == len(mod_data)
        ori_sentences = []
        mod_sentences = []
        flag_list = list(np.ones((len(ori_data))))
        for i in range(len(ori_data)):
            ori_sentences.append(clean_text(ori_data[i]["pred_output"]))
            mod_sentences.append(clean_text(mod_data[i]["pred_output"]))
            if clean_text(ori_data[i]["pred_output"]).strip()=='' or clean_text(mod_data[i]["pred_output"]).strip()=='':
                flag_list[i] = 0

        embeddings1 = emb_model.encode(ori_sentences)
        embeddings2 = emb_model.encode(mod_sentences)

        similarities = util.pytorch_cos_sim(embeddings1, embeddings2)
        sim = torch.diag(similarities)
        flag_tensor = torch.tensor(flag_list)
        result = sim * flag_tensor

        for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            value=cal_above_thresh(result, thresh)
            sim_thre_list.append({"model":model, "type":type, "thresh":thresh, "value":value})

df = pd.DataFrame(sim_thre_list)
csv_file="ectsum_sim_clean_draw.csv"
df.to_csv(csv_file, index=False)

