import json
from collections import Counter
import json
from typing import List, Dict, Any
import ast
import regex
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import numpy as np
import string
from nltk.translate.meteor_score import meteor_score
import os
    
from vllm_client import VllmAsyncClient, load_all_lora_adapters, unload_all_lora_adapters
import json
import os
import asyncio
from datasets import load_dataset
from prompts import get_qa, get_all_qa, get_suggestion_prompt, get_suggestion_prompt_wo_oa

HOME_PATH = os.getenv("HOME_PATH", ".")
API_URL = "http://localhost:8000/v1/chat/completions"


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


async def main():

    client = VllmAsyncClient(
        base_model_name=MODEL_NAME,
    )

    with open(os.path.join(CONFIG_PATH, f"task.json"), "r") as fin:
        dataset_config = json.load(fin)
    if DATA_TYPE not in dataset_config:
        with open(os.path.join(CONFIG_PATH, f"domain.json"), "r") as fin:
            dataset_config = json.load(fin)
    
    for item in dataset_config[DATA_TYPE]:
        DATASET_NAME = item["dataset_name"]
        if "Locomo" in DATASET_NAME or "DialSim" in DATASET_NAME:
            continue
        MAX_TOKENS = item["max_output_len"] if item["max_output_len"] is not None else 2048
        MAX_TOKENS = max(MAX_TOKENS, 2048)
        
        dataset = json.load(open(os.path.join(DATA_PATH, DATASET_NAME, "with_feedback_rules.json"), "r"))
        for i in range(len(dataset)):
            dataset[i]["user_prompt"] = dataset[i].pop("prompt")
            if dataset[i]["lang"] == "zh":
                dataset[i]["feedback"] = dataset[i]["user_prompt"][-1]["content"].split("\n\n## 用户反馈\n")[-1].strip()
            else:
                dataset[i]["feedback"] = dataset[i]["user_prompt"][-1]["content"].split("\n\n## User Feedback\n")[-1].strip()
            dataset[i]["suggestion_response"] = dataset[i].pop("_raw_response")
        
        
        questions = []
        new_results = []
        for item in dataset:
            if type(item["suggestion_response"]) == str:
                continue
            rules = item["suggestion_response"].get("rules", [])
            if not rules:
                new_item = item.copy()
                new_item["new_a"] = [item["old_a"]]
                new_results.append(new_item)
                continue
            for REVISE_TYPE in REVISE_TYPES:
                if REVISE_TYPE == "rulesandoa":
                    revise_prompt = get_suggestion_prompt(rules=rules, lang=item["lang"], old_answer=item["old_a"], q=item["q"])
                elif REVISE_TYPE == "rulesandoa_v2":
                    revise_prompt = get_suggestion_prompt(rules=rules, lang=item["lang"], old_answer=item["old_a"], q=item["q"], version="v2")
                elif REVISE_TYPE == "rulesandfeedbackandoa_v2":
                    combined_rules = rules + [item["feedback"]]
                    revise_prompt = get_suggestion_prompt(rules=combined_rules, lang=item["lang"], old_answer=item["old_a"], q=item["q"], version="v2")
                elif REVISE_TYPE == "rules":
                    revise_prompt = get_suggestion_prompt_wo_oa(rules=rules, lang=item["lang"], q=item["q"])
                else:
                    revise_prompt = get_suggestion_prompt_wo_oa(rules=item["feedback"], lang=item["lang"], q=item["q"])
                questions.append({
                    "test_idx": item["test_idx"],
                    "prompt": revise_prompt,
                    "old_a": item["old_a"],
                    "lang": item["lang"],
                    "q": item["q"],
                    "user_prompt": item["user_prompt"],
                    "suggestion_response": item["suggestion_response"]
                })
        for j in range(GENERATE_NUM):
            results = await client.generate_batch(questions, generation_config_overrides={
                "max_tokens": MAX_TOKENS,
                "temperature": 1.0,
                "top_p": 0.9,
                "seed": j + 42,
            })
            if j == 0:
                for i in range(len(results)):
                    new_item = results[i].copy()
                    new_item["new_a"] = [results[i]["_raw_response"]]
                    new_results.append(new_item)
            else:
                for i in range(len(results)):
                    for nr in new_results:
                        if nr["test_idx"] == results[i]["test_idx"]:
                            nr["new_a"].append(results[i]["_raw_response"])
                            break
            
        
        os.makedirs(os.path.join(OUTPUT_PATH, DATASET_NAME), exist_ok=True)
        with open(os.path.join(OUTPUT_PATH, DATASET_NAME, "answers.json"), "w") as fout:
            json.dump(new_results, fout, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="quick start")
    parser.add_argument("--model", type=str, default="Qwen3-8B", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--output_root_path", type=str, default="./UNO/code/output", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--config_path", type=str, default="path/to/configs/datasets", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--data_type", type=str, default="Long-Short", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--generate_num", type=int, default=5, help="number of suggestions to generate")
    parser.add_argument("--rules_dir_name", type=str, default="rules")
    parser.add_argument("--output_dir_name", type=str, default="revised_answers")
    parser.add_argument("--revise_type", type=str, default="4", help="revise type: with_feedback / no_feedback")
    
    args = parser.parse_args()
    global MODEL_NAME, OUTPUT_PATH, CONFIG_PATH, DATA_TYPE, DATA_PATH, GENERATE_NUM, REVISE_TYPES
    type_list = ["feedback", "rules", "rulesandoa_v2", "rulesandfeedbackandoa_v2"]
    REVISE_TYPES = [type_list[int(rt) - 1] for rt in args.revise_type.split("_")]
    GENERATE_NUM = args.generate_num
    CONFIG_PATH = os.path.join(HOME_PATH, args.config_path)
    MODEL_NAME = args.model 
    DATA_TYPE = args.data_type

    OUTPUT_PATH = os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, args.output_dir_name)
    DATA_PATH = os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, args.rules_dir_name)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    asyncio.run(main())
