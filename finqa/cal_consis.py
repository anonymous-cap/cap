import os
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




