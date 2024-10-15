import os
import json


def load_json(file):
    with open(file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def jaccard_similarity(string1, string2):

    set1 = set(string1.split(' '))
    set2 = set(string2.split(' '))

    intersection = set1.intersection(set2)
    union = set1.union(set2)


    if len(union) == 0:
        return 0


    return len(intersection) / len(union)



for model in ["llama", "mistral", "finma_nlp", "finma_full", "fingpt","gpt","baichuan","disc_fin"]:
    pred_dir = f"{model}_pred"
    print(model)
    for type in ['train','eval']:
        ori_file=f"research_ori_{type}.json"
        mod_file=f"research_mod_{type}.json"
        ori_data = load_json(os.path.join(pred_dir,ori_file))
        mod_data= load_json(os.path.join(pred_dir,mod_file))
        assert len(ori_data)==len(mod_data)
        print(len(ori_data))
        jar_list=[]
        for i in range(len(ori_data)):
            if ("what is the diluted earnings per sharein 2014?" not in ori_data[i]['input']) and ("income taxes have not been provided is approximately" not in ori_data[i]['input']) and ("stock-based awards under the plan stock options 2013 marathon grants" not in ori_data[i]['input']) and ("written trading plans that comply with rule 10b5-1" not in ori_data[i]['input']) and ("software and will give the company a comprehensive design-to-silicon flow" not in ori_data[i]['input']):
                ori_item=ori_data[i]['pred_output']
                mod_item=mod_data[i]['pred_output']
                jar= jaccard_similarity(ori_item.strip(),mod_item.strip())
                jar_list.append(jar)
        print(type, sum(jar_list)/len(jar_list))
    print()



