import argparse
import os
import ast
import json
from datasets import load_dataset
from prompts import get_all_qa
import subprocess
import numpy as np
from time import sleep

HOME_PATH = os.getenv("HOME_PATH", ".")
LLM_PATH = os.getenv("LLM_PATH", "../LLM")

def convert_str_to_obj(example):
    for col in example.keys():
        if col.startswith("dialog") or col.startswith("implicit_feedback") or col in ["input_chat_messages", "info"]:
            if isinstance(example[col], str):
                try:
                    example[col] = ast.literal_eval(example[col])
                except (ValueError, SyntaxError):
                    try:
                        example[col] = json.loads(example[col])
                    except Exception:
                        pass
    if "Locomo" in example["dataset_name"]:
        if example["info"]["category"] == 5:
            example["info"]["golden_answer"] = json.dumps(example["info"]["golden_answer"])
        else:
            example["info"]["golden_answer"] = str(example["info"]["golden_answer"])
    return example

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import jieba
import re

def smart_tokenize(text):
    if not isinstance(text, str):
        return []
    contains_chinese = bool(re.search(r'[\u4e00-\u9fa5]', text))
    if contains_chinese:
        return list(jieba.cut(text))
    else:
        return text.strip().lower().split()

def calculate_metrics(reference_text, output_text):
    tok_output = smart_tokenize(output_text)
    tok_ref = smart_tokenize(reference_text)
    cc = SmoothingFunction()
    try:
        if len(tok_ref) == 0 or len(tok_output) == 0:
            bleu4 = 0.0
        else:
            bleu4 = sentence_bleu([tok_ref], tok_output, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method1)
    except:
        bleu4 = 0.0
    return bleu4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen3-8B", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--small_test", action="store_true", help="Use a small test dataset for quick debugging.")
    parser.add_argument("--root_path", type=str, default="./UNO/code/output2", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--data_type", type=str, default="Long-Short", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--cluster_dir_name", type=str, default="cluster")
    parser.add_argument("--rules_dir_name", type=str, default="rules")
    parser.add_argument("--train_output_dir", type=str, default="rule_train", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--lr", type=str, default="5e-4", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--config_path", type=str, default="path/to/configs/datasets", help="model name used in your deployment/model service endpoint")
    args = parser.parse_args()
    CONFIG_PATH = os.path.join(HOME_PATH, args.config_path)
    DATA_TYPE = args.data_type
    ROOT_PATH = os.path.join(HOME_PATH, args.root_path, DATA_TYPE)
    CLUSTER_DIR = os.path.join(ROOT_PATH, args.cluster_dir_name)
    RULES_DIR = os.path.join(ROOT_PATH, args.rules_dir_name)
    with open(os.path.join(CLUSTER_DIR, "cluster_results.json"), "r") as fin:
        cluster_results = json.load(fin)
    subprocess.run(f"python get_rule_train_data.py {RULES_DIR} {str(args.small_test)}", shell=True, check=True)
    for cluster_key in cluster_results.keys():
        if not os.path.exists(os.path.join(CLUSTER_DIR, args.train_output_dir, f'{cluster_key}_{args.model}', "adapter_model.safetensors")):
            print(f"Start training for cluster {cluster_key}...")
            try:
                subprocess.run(f"bash train/sft.sh {RULES_DIR} {cluster_key} {os.path.join(CLUSTER_DIR, 'cluster_results.json')}                  4 {HOME_PATH} {LLM_PATH} {os.path.join(CLUSTER_DIR, args.train_output_dir)} {args.model}", shell=True, check=True)
            except:
                pass
            sleep(20)
    print("All clusters have been trained. Starting Test inference...")
    if args.small_test:
        subprocess.run(f"python get_revise_test.py --model {args.model} --small_test --data_type {args.data_type} --output_file_name test_results_reverse_small.json --cluster_file_abs_path {CLUSTER_DIR} --lora_path {os.path.join(CLUSTER_DIR, args.train_output_dir)}", shell=True, check=True)
    else:
        subprocess.run(f"python get_revise_test.py --model {args.model} --data_type {args.data_type} --output_file_name test_results_reverse.json --cluster_file_abs_path {CLUSTER_DIR} --lora_path {os.path.join(CLUSTER_DIR, args.train_output_dir)}", shell=True, check=True)
    print("Test inference completed.")
