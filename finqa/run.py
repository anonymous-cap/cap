import argparse
import os
import sys

from utils.utils_llama import run as run_llama
from utils.utils_mistral import run as run_mistral
from utils.utils_baichuan import run as run_baichuan
from utils.utils_fingpt import run as run_fingpt
#from utils.utils_gpt import run as run_gpt
def main(args):
    # setting model
    model = args.model
    model_dict = {
        "llama": "Meta-Llama-3-8B-Instruct-hf",
        "finma_full": "finma-7b-full/",
        "finma_nlp": "finma-7b-nlp/",
        "fingpt":"llama-2-7b-hf",
        "mistral": "Mistral-7B-Instruct-v0.3",
        "baichuan": "Baichuan-13B-Chat",
        "disc_fin": "DISC-FinLLM/",
        "gpt":"gpt-4o-mini-2024-07-18"
    }
    model_path = model_dict[model]

    if model.startswith('llama') or model.startswith('finma'):
        run_function = run_llama
    elif model == 'mistral':
        run_function = run_mistral
    elif model == "baichuan" or model == "disc_fin":
        run_function = run_baichuan
    elif model=="fingpt":
        run_function = run_fingpt
    # elif model=="gpt":
    #     run_function = run_gpt
    else:
        raise NotImplementedError("Run function for model '{}' is not implemented".format(model))

    out_dir = f"{model}_pred"

    # setting data/task
    task = args.task
    task_dict = {"fineval": ["ori_dev.json", "mod_dev.json", "ori_val.json", "mod_val.json"],
                 "finqa": ["ori_train_samples.json", "mod_train_samples.json", "ori_dev_samples.json", "mod_dev_samples.json", "ori_test_samples.json",
                           "mod_test_samples.json"],
                 "alphafin": ["report_ori_train.json", "report_mod_train.json", "research_ori_train.json",
                              "research_mod_train.json", "research_ori_eval.json", "research_mod_eval.json"],
                 "ectsum": ["ori_train.json", "mod_train.json", "ori_val.json", "mod_val.json", "ori_test.json",
                            "mod_test.json"]}
    for file in task_dict[task]:
        file_path = os.path.join("data", file)
        run_function(model_path, file_path, file, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run models with specified parameters.")
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--task', type=str, default="finqa", help='Task name')
    args = parser.parse_args()
    main(args)