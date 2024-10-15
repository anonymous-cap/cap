import json


def load_json(file):
    with open(file, 'r') as f:
        data=json.load(f)
    print(file, "with data length: ", len(data))
    return data