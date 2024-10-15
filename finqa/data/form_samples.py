import os
import json
import random

def load_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(file, "with data length: ", len(data))
    return data

def save_json(data,out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

random.seed(42)
tmp=list(range(500))
random.shuffle(tmp)

data_dir="."
files=["ori_train.json", "ori_dev.json", "ori_test.json", "mod_train.json", "mod_dev.json", "mod_test.json"]
for file in files:
    file_path=os.path.join(data_dir, file)
    data=load_json(file_path)
    data_samples=[data[i] for i in tmp if data[i]["output"]]
    file_name=file.replace('.json','')
    save_json(data_samples[:100],os.path.join(data_dir, file_name+"_samples.json"))




