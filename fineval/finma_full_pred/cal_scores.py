import argparse
import os
import sys
sys.path.append('/Users/yizhao/Projects/FDC/fdc')
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

# surf_eval=SurfaceEval()
# for file in ["mod_dev.json", "mod_val.json","ori_dev.json","ori_val.json"]:
#     data=load_json(file)
#     f1_list=[]
#     em_list=[]
#     fact_em_list=[]
#     for item in data[:170]:
#         ground=item['output'].strip()
#         pred=item['pred_output'].strip()
#
#         if pred==ground:
#             em_list.append(1)
#         else:
#             em_list.append(0)
#
#
#     print()
#     print("File: ", file)
#     print("Data Length: ", len(data))
#     print("Mean EM: ", round(sum(em_list)/len(em_list),4))
#     # print("Mean Correctness: ", round(sum(fact_em_list)/len(fact_em_list),4))
#     # print("------------------------------------------------")


# File:  mod_dev.json
# Data Length:  170
# Mean EM:  0.1
#
# File:  mod_val.json
# Data Length:  1151
# Mean EM:  0.1647
#
# File:  ori_dev.json
# Data Length:  170
# Mean EM:  0.1529
#
# File:  ori_val.json
# Data Length:  1151
# Mean EM:  0.1706
