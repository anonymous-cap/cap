import json

def load_json(file):
    with open(file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def jaccard_similarity(string1, string2):
    set1, set2 = set(string1), set(string2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)


    if len(union) == 0:
        return 0

    return len(intersection) / len(union)



for type in ['train']:
    file_ori=f"ori_{type}.json"
    file_mod=f"mod_{type}_samples.json"


    data_ori=load_json(file_ori)
    data_mod=load_json(file_mod)

    cnt=0
    jar_list=[]
    for i in range(len(data_ori)):
        inp_i=data_ori[i]['output']
        for j in range(len(data_mod)):
            inp_j=data_mod[j]['output']
            if inp_i==inp_j:
                cnt+=1
                ori_pred = data_ori[i]['pred_output']
                mod_pred = data_mod[j]['pred_output']
                jar = jaccard_similarity(ori_pred, mod_pred)
                jar_list.append(jar)

                break
    print(cnt)
    print(sum(jar_list)/len(jar_list))
