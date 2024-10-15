import os
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from .op import load_json
import json

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.float16, device_map="auto")
    model.model_parallel = True
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    peft_model = "FinGPT/fingpt-mt_llama2-7b_lora"
    model = PeftModel.from_pretrained(model, peft_model)
    model = model.eval()
    return model, tokenizer


def infer_instruction(model, tokenizer,instruction):
    inputs = tokenizer(instruction, return_tensors='pt',return_token_type_ids=False)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output = model.generate(**inputs,  max_new_tokens=1024,do_sample=False,eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def run(model_path, file_path, out_file, out_dir="fingpt_pred"):
    model, tokenizer = load_model(model_path)
    data=load_json(file_path)
    result=[]
    for item in tqdm(data):
        instruction=item["input"]
        output=item["output"]
        pred_output=infer_instruction(model, tokenizer, instruction)
        result.append({"input":instruction, "output": output, "pred_output":pred_output.replace(instruction, '')})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{out_file}"), 'w', encoding='utf-8') as f:
        json.dump(result,f, ensure_ascii=False, indent=2)
    print(file_path, "Done!")