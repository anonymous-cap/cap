import os
import torch
from tqdm import tqdm
from .op import load_json
import json
import random
from openai import OpenAI


def infer_instruction(model,prompt):
  api_keys = os.environ.get("OPENAI_API_KEYS")
  api_key_list = api_keys.split(',')
  api_key = random.sample(api_key_list, 1)[0]
  client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))
  completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt} ],
    temperature=0.0,
  )
  response=completion.choices[0].message.content
  if response:
    return response
  else:
    return "Please run again."

def run(model,file_path, out_file, out_dir="gpt_4o_mini_pred"):
    data=load_json(file_path)
    result=[]
    for item in tqdm(data):
        instruction=item["input"]
        output=item["output"]
        response=infer_instruction(model, instruction)
        result.append({"input":instruction, "output": output, "pred_output":response})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{out_file}"), 'w', encoding='utf-8') as f:
        json.dump(result,f, ensure_ascii=False, indent=2)
    print(file_path, "Done!")