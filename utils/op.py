import json


def load_json(file):
    with open(file, 'r') as f:
        data=json.load(f)
    #print(file, "with data length: ", len(data))
    return data



def save_json(data, out_file):
    with open(out_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return None