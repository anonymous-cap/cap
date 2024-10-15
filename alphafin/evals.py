import os
import csv
import json
import evaluate






#load for english text processing
rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")


def load_json(file):
    with open(file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def cal_surf_scores(pred, ref):
    if not pred:
        bleu_score=0
    else:
        bleu_res=bleu.compute(predictions=[pred],references=[ref])
        bleu_score=bleu_res['bleu']
    rouge_res = rouge.compute(predictions=[pred],references=[ref])
    rouge1=rouge_res['rouge1']
    rouge2=rouge_res['rouge2']
    rougeL=rouge_res['rougeL']
    return bleu_score,rouge1,rouge2,rougeL



csv_file = "results.csv"

with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)


    writer.writerow(["Model", "File Path", "Data Length", "BLEU Avg", "ROUGE-1 Avg", "ROUGE-2 Avg", "ROUGE-L Avg"])


    for model in ["llama", "mistral", "finma_nlp", "finma_full", "fingpt", "baichuan", "disc_fin", "gpt"]:
        pred_dir = f"{model}_pred"
        for file in ['research_ori_train.json', 'research_mod_train.json', 'research_ori_eval.json', 'research_mod_eval.json']:
            file_path = os.path.join(pred_dir, file)
            data = load_json(file_path)

            bl_list = []
            rg1_list = []
            rg2_list = []
            rgl_list = []


            for item in data:
                ref = item['output']
                pred = item['pred_output']
                bl, rg1, rg2, rgl = cal_surf_scores(pred, ref)
                bl_list.append(bl)
                rg1_list.append(rg1)
                rg2_list.append(rg2)
                rgl_list.append(rgl)


            bl_avg = sum(bl_list) / len(bl_list)
            rg1_avg = sum(rg1_list) / len(rg1_list)
            rg2_avg = sum(rg2_list) / len(rg2_list)
            rgl_avg = sum(rgl_list) / len(rgl_list)


            writer.writerow([model, file_path, len(data), bl_avg, rg1_avg, rg2_avg, rgl_avg])

print(f"Results saved to {csv_file}")