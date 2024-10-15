import argparse
import os
import sys
from tqdm import tqdm

from utils.evals import SurfaceEval, LLMEval
from utils.op import load_json, save_json

import re


def extract_options(text):
    pattern = r"选项：(.*?)\n 答案"
    match = re.search(pattern, text, re.S)
    if match:
        return match.group(1)
    else:
        return None

surf_eval=SurfaceEval()
llm_eval=LLMEval("gpt-4o")
for pred_dir in ["llama_pred", "mistral_pred","finma_nlp_pred","finma_full_pred"]:
    for file in [ "ori_val.json"]:
        print(file)
        out_dir=os.path.join(pred_dir, "scores")
        os.makedirs(out_dir, exist_ok=True)
        data=load_json(os.path.join(pred_dir, file))
        data_with_score=[]
        fail_cnt=0
        for item in tqdm(data):
            try:
                ground=item["output"]
                pred=item["pred_output"]
                f1,em=surf_eval.get_f1_e(pred, ground)
                print(f1, em)
                input=item["input"]
                options=extract_options(input)
                gpt_score=llm_eval.fineval_score(options, ground, pred)
                print(gpt_score)
                data_with_score.append({"input":item["input"], "output":item["output"], "pred_output":item["pred_output"], "scores":[f1, em, gpt_score]})
            except:
                fail_cnt+=1
        save_json(data_with_score, os.path.join(out_dir, file))
        print(fail_cnt)

