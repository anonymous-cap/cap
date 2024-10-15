import os
import re
import json
import requests
import pdb
from tqdm import tqdm


def load_json(file):
    with open(file, "r") as f:
        return json.load(f)



def get_li(inp):
    pattern = r"选项：(\[.*?\])"
    match = re.search(pattern, inp)
    return match.group(1)

def run_paraphrase(file_path):
    data_up=[]
    keys = os.getenv('CLAUDE_KEYS').split(',')
    Baseurl = "https://api.claude-Plus.top"
    data=load_json(file_path)
    prompt_temp = """Input: A string and a list containing options A, B, C, D with corresponding content.
    Task: Determine which option (A, B, C, D) the string corresponds to in the list.
    Output Requirements:

    1. If a match is found, output the corresponding letter (A, B, C, or D).
    2. If no match can be determined, output an empty string "".
    3. Provide only the output letter or empty string, with no additional words.
    4. Remember output 'A', 'B', 'C', or, 'D'!!!!it's very important!!! and you need to make sure your answer is correct!!!

    Examples:

    Input:
    String: "增加"
    List: ['A':增加或减少无法确定, 'B': 不变, 'C':减少, 'D':增加]
    Output: D
    Input:
    String: " 'C': 进行玉米期转现交易"
    List: ['A':买入玉米期货合约, 'B': 卖出玉米期货合约, 'C':进行玉米期转现交易, 'D':进行玉米期现套利]
    Output: C
    Input:
    String: "['A': 障碍期权, 'B': 回望期权, 'C': 亚式期权, 'D': 彩虹期权]"
    List: ['A':障碍期权, 'B': 回望期权, 'C':亚式期权, 'D':彩虹期权]
    Output: ""
    Input:
    String: "37000.0"
    List: ['A':-20000美元, 'B': 20000美元, 'C':-37000美元, 'D':37000美元]
    Output: ""
    Now please answer this question:

    Input: String: {}, List: {}
    Output:
    """
    for i in tqdm(range(len(data))):
        item=data[i]
        li=get_li(item["input"])
        st=item["pred_output"].strip()
        if st in ["A","B","C","D"]:
            data_up.append({"input":item["input"],"output":item["output"],"pred_opt":st,"pred_output":item["pred_output"]})
            continue

        prompt = prompt_temp.format(st, li)
        Skey = keys[i%3]
        payload = json.dumps({
            "model": "claude-3-5-sonnet-20240620",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        url = Baseurl + "/v1/chat/completions"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {Skey}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        resp = response.json()
        if resp['choices'][0]['message']['content']:
            resp_content = resp['choices'][0]['message']['content']
            data_up.append({"input":item["input"],"output":item["output"], "pred_opt":resp_content,"pred_output":item["pred_output"]})
        else:
            print(i)
            print(item)
        # except:
        #     item_up = {"input": item["input"], "output": item["output"], "pred_output": item["pred_output"],
        #                "pred_opt": st}
        #     print(i)

    return data_up




for dir_path in ["../finma_full_pred/post_human", "../finma_nlp_pred/post_human"]:
    for ty in ["ori", "mod"]:
        file_path=os.path.join(dir_path,f"post_{ty}_dev.json")
        data_up=run_paraphrase(file_path)
        out_file=os.path.join(dir_path,f"post_post_{ty}_dev.json")
        with open(out_file, "w") as f:
            json.dump(data_up, f,ensure_ascii=False,indent=2)
        print(f"PLEASE SEE: {out_file}")

for dir_path in ["../finma_full_pred/post_human"]:
    for ty in ["ori"]:
        file_path=os.path.join(dir_path,f"post_{ty}_val.json")
        data_up=run_paraphrase(file_path)
        out_file=os.path.join(dir_path,f"post_post_{ty}_val.json")
        with open(out_file, "w") as f:
            json.dump(data_up, f,ensure_ascii=False,indent=2)
        print(f"PLEASE SEE: {out_file}")




