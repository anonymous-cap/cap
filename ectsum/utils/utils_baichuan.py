# -- coding: utf-8 --
import os
import json
import torch
from .op import load_json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.max_new_tokens = 1024
    model.generation_config.do_sample = False
    return model, tokenizer

def infer_instr(model, tokenizer, instr):
    messages = [{"role": "user", "content": instr}]
    response = model.chat(tokenizer, messages)
    return response

def run(model_path, file_path, out_file, out_dir="baichuan_pred"):
    model, tokenizer = load_model(model_path)
    data=load_json(file_path)
    result=[]
    for item in tqdm(data):
        instr=item["input"]
        output=item["output"]
        pred_output=infer_instr(model, tokenizer, instr)
        result.append({"input":instr, "output": output, "pred_output":pred_output})
        torch.cuda.empty_cache()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{out_file}"), 'w', encoding='utf-8') as f:
        json.dump(result,f, ensure_ascii=False, indent=2)
    print(file_path, "Done!")


