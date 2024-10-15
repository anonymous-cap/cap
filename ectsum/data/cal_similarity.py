import json
import os
import random
import pdb
from sentence_transformers import util
import pickle
import torch
from transformers import LongformerForMaskedLM, LongformerTokenizer
from tqdm import tqdm

random.seed(42)
def get_text_emb(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt",padding="max_length", truncation=True,max_length=4096)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]
    emb= last_hidden_state.mean(dim=1)
    return emb

def save_json(data,out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
    return None

def get_ori_mod_data(type_k, N):
    with open(f"ori_{type_k}.json", "r") as f:
        ori_data = json.load(f)
    with open(f"mod_{type_k}.json", "r") as f:
        mod_data = json.load(f)

    ids = random.sample(range(len(ori_data)), N)
    ori_samples=[ori_data[i] for i in ids]
    mod_samples=[mod_data[i]  for i in ids]
    return ori_samples, mod_samples


def cal_similarity(ori_samples, mod_samples, model, tokenizer):
    sim_list=[]
    for ori_sample, mod_sample in tqdm(zip(ori_samples, mod_samples)):
        ori_input=ori_sample["input"]
        mod_input=mod_sample["input"]
        ori_text=ori_input.replace("Please summarize the content of the context into bullet points and provide only the bullet point summary.\nContext:","").replace("\nSummary: ","")
        mod_text=mod_input.replace("Please summarize the content of the context into bullet points and provide only the bullet point summary.\nContext:","").replace("\nSummary: ","")
        ori_emb =get_text_emb(ori_text, model, tokenizer)
        mod_emb =get_text_emb(mod_text, model, tokenizer)
        sim = util.cos_sim(ori_emb, mod_emb).item()
        sim_list.append(sim)
    return sim_list



if __name__=="__main__":
    for type_k in ["train", "val", "test"]:

        model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        ori_samples, mod_samples=get_ori_mod_data(type_k, 100)
        save_json(ori_samples, f"ori_{type_k}_samples.json")
        save_json(mod_samples, f"mod_{type_k}_samples.json")
        sim_list=cal_similarity(ori_samples, mod_samples, model, tokenizer)

        with open(f'sim_dict_{type_k}.pkl', 'wb') as f:
            pickle.dump(sim_list, f)

        min_s = round(min(sim_list), 4)
        max_s = round(max(sim_list), 4)
        mean_s = round(sum(sim_list) / len(sim_list), 4)
        print(type_k, len(sim_list))
        print(min_s, max_s, mean_s)

