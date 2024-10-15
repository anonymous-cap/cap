import json
import os
import re

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .op import load_json


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).half()
    model.to("cuda")
    return model, tokenizer


def infer_instr(model, tokenizer, instr):
    messages = [{"role": "user", "content": instr}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to("cuda")
    generated_ids = model.generate(model_inputs, max_new_tokens=1024, do_sample=False)
    decoded = tokenizer.batch_decode(generated_ids)
    response = decoded[0]
    return response


def response_post_processing(text):
    pattern = re.compile(r'\[/INST\](.*?)</s>', re.DOTALL)
    match = pattern.search(text)
    res = ''
    if match:
        res = match[0].replace('[/INST]', '').replace('</s>', '').strip()
    else:
        if '[/INST]' in text:
            res = text.split('[/INST]')[-1].strip()
    return res


def run(model_path, file_path, out_file, out_dir="mistral_pred"):
    model, tokenizer = load_model(model_path)
    data = load_json(file_path)
    result = []
    for item in tqdm(data):
        instr = item["input"]
        output = item["output"]
        pred_output = infer_instr(model, tokenizer, instr)
        result.append({"input": instr, "output": output, "pred_output": response_post_processing(pred_output)})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{out_file}"), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(file_path, "Done!")
