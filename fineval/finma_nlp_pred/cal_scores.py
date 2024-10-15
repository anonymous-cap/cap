import argparse
import os
import sys

from utils.evals import SurfaceEval, LLMEval
from utils.op import load_json, save_json


surf_eval=SurfaceEval()
for file in ["mod_dev.json", "mod_val.json","ori_dev.json","ori_val.json"]:
    data=load_json(os.path.join("post_human","post_"+file))
    f1_list=[]
    em_list=[]
    fact_em_list=[]
    for item in data:
        ground=item['output'].strip()
        pred=item['pred_output'].strip()
        pred_fact = item["pred_opt"].strip()


        #f1_list.append(surf_eval.f1_score(pred, ground))
        em_list.append(surf_eval.exact_match_score(pred, ground))
        # fact_em_list.append(surf_eval.exact_match_score(pred_fact, ground))

    print()
    print("File: ", file)
    print("Data Length: ", len(data))
    # print("Mean F1: ", round(sum(f1_list)/len(f1_list),4))
    print("Mean EM: ", round(sum(em_list)/len(em_list),4))
    # print("Mean Correctness: ", round(sum(fact_em_list)/len(fact_em_list),4))
    # print("------------------------------------------------")
