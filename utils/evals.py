from collections import Counter
import string
import re
import os
import random
from openai import OpenAI
import requests
import json

class SurfaceEval():
    def __init__(self):
        pass

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self,pred,ground):

        prediction_tokens = self.normalize_answer(pred).split()
        ground_truth_tokens = self.normalize_answer(ground).split()

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)

        num_same = sum(common.values())
        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(prediction_tokens)

        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self,pred,ground):
        if self.normalize_answer(pred) == self.normalize_answer(ground):
            return 1
        else:
            return 0

    def get_f1_e(self,pred, ground):
        f1=self.f1_score(pred,ground)
        em=self.exact_match_score(pred,ground)
        return f1,em


class LLMEval():
    def __init__(self, model):
        self.model=model

    def infer_gpt(self, prompt):
        api_keys = os.environ.get("OPENAI_API_KEYS")
        api_key_list = api_keys.split(',')
        api_key = random.sample(api_key_list, 1)[0]
        client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))
        completion = client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
        response = completion.choices[0].message.content
        if response:
            return response
        else:
            return ""



    def fineval_score(self, options, ground, pred):
        prompt=f"""
        I will provide you with two strings, one is the prediction, and the other is the ground truth.
        The ground truth will be one of the options A, B, C, or D, each corresponding to specific content.
        You need to determine whether the prediction is factually correct. 
        For example, if the prediction is in lowercase, contains the specific content of the option, or doesn't explicitly mention A, B, C, or D but conveys the specific content, the prediction is still consistent with the ground truth.
        Finally, you only need to output whether they are consistent. 
        
        The option list is: {options};
        ground truth is: {ground};
        prediction string is: {pred};
        Are they consistent? 
        If they are consistent, please output ‘1’; if not, output ‘0’. 
        Please only output ‘0’ or ‘1’, without any additional words.
        """
        return self.infer_gpt(prompt)


