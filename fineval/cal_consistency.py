import os
import re
import json

import pdb

def load_json(file):
    with open(file, "r") as f:
        data=json.load(f)
    return data


def extract_options(text):
    pattern = r"'A':(.*?), 'B':(.*?), 'C':(.*?), 'D':(.*?)\]"


    match = re.search(pattern, text)

    if match:

        options = [match.group(1), match.group(2), match.group(3), match.group(4)]
        return [opt.strip() for opt in options]
    else:
        return None


def cal_consistency(file_ori, file_mod):
    data_ori = load_json(file_ori)
    data_mod = load_json(file_mod)

    cnt = 0
    opt_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    for i, item in enumerate(data_ori):
        item_ori = data_ori[i]
        item_mod = data_mod[i]

        fact_opt_ori = item_ori["pred_opt"]
        fact_opt_mod = item_mod["pred_opt"]
        if fact_opt_ori in ["A", "B", "C", "D"] and fact_opt_mod in ["A", "B", "C", "D"]:
            ori_options = extract_options(item_ori["input"])
            mod_options = extract_options(item_mod["input"])
            if ori_options[opt_map[fact_opt_ori]] == mod_options[opt_map[fact_opt_mod]]:
                cnt += 1
    return cnt/len(data_ori)

for model in ["baichuan", "disc_fin","gpt","mistral","llama"]:
    pred_dir=f"{model}_pred/post_human/"
    print(pred_dir)
    file_pairs=[["post_ori_dev.json","post_mod_dev.json"], ["post_ori_val.json","post_mod_val.json"]]
    for pair in file_pairs:
        file_ori=os.path.join(pred_dir,pair[0])
        file_mod=os.path.join(pred_dir,pair[1])
        con=cal_consistency(file_ori, file_mod)
        print(pair)
        print("Consistency: ",round(con,4))
        print()


for model in ["finma_full","finma_nlp"]:
    pred_dir=f"{model}_pred/post_human/"
    print(pred_dir)
    file_pairs=[["post_post_ori_dev.json","post_post_mod_dev.json"], ["post_post_ori_val.json","post_post_mod_val.json"]]
    for pair in file_pairs:
        file_ori=os.path.join(pred_dir,pair[0])
        file_mod=os.path.join(pred_dir,pair[1])
        con=cal_consistency(file_ori, file_mod)
        print(pair)
        print("Consistency: ",round(con,4))
        print()


