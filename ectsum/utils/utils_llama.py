import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from .op import load_json
import json

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16).cuda()
    tokenizer.use_default_system_prompt = False
    return model, tokenizer


def infer_instruction(model, tokenizer,instruction):
    input_ids = tokenizer.encode(instruction, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    output = model.generate(input_ids,  max_new_tokens=1024, do_sample=False)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def run(model_path, file_path, out_file, out_dir="llama_pred"):
    model, tokenizer = load_model(model_path)
    data=load_json(file_path)
    result=[]
    for item in tqdm(data):
        try:
            instruction=item["input"]
            output=item["output"]
            pred_output=infer_instruction(model, tokenizer, instruction)
            result.append({"input":instruction, "output": output, "pred_output":pred_output.replace(instruction, '')})
        except:
            print("ERROR")
        torch.cuda.empty_cache()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{out_file}"), 'w', encoding='utf-8') as f:
        json.dump(result,f, ensure_ascii=False, indent=2)
    print(file_path, "Done!")