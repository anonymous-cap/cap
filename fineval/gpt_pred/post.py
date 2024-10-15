import json
import re

def load_json(file):
    with open(file, 'r') as f:
        data=json.load(f)
    #print(file, "with data length: ", len(data))
    return data

def save_json(data, out_file):
    with open(out_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return None




data=load_json("mod_val.json")
data_up=[]
for item in data:
    pred=item["pred_output"]
    tmp=pred.strip().split('\n')
    opt=re.sub(r'[^A-Z]', '', tmp[-1])
    data_up.append({"input":item["input"], "output":item["output"], "pred_opt":opt,"pred_output":item["pred_output"] })
save_json(data_up, "post_human/post_mod_val.json")


