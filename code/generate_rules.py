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
from prompts import get_qa, get_all_qa, get_prompt, get_prompt_no_feedback, rules_schema

HOME_PATH = os.getenv("HOME_PATH", ".")
MEMORYBENCH_PATH = os.getenv("MEMORYBENCH_PATH")
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
        if MODEL_NAME == "Qwen3-8B":
            if not MEMORYBENCH_PATH:
                raise ValueError("Please set MEMORYBENCH_PATH to the local MemoryBench dataset path.")
            dataset = load_dataset(MEMORYBENCH_PATH, DATASET_NAME)
            dataset = dataset.map(convert_str_to_obj)["train"]
        else:
            dataset = json.load(open(os.path.join(f"../../../MemoryBench/dialogs-{MODEL_NAME}", DATASET_NAME, "wo_memory/dataset.json"), "r"))
            dataset = dataset["train"]
        qa = get_all_qa(dataset)
        qaf = get_qa(dataset)
        
        questions_no_feedback = [{
            "test_idx": idx,
            "prompt": get_prompt_no_feedback(q, lang),
            "old_a": a,
            "lang": lang,
            "q": q
        } for q, a, lang, idx in qa]
        
        questions_with_feedback = [{
            "test_idx": idx,
            "prompt": get_prompt(q, a, f, lang),
            "old_a": a,
            "lang": lang,
            "q": q
        } for q, a, f, lang, idx in qaf]
        
        results_no_feedback = await client.generate_batch(questions_no_feedback, json_format=rules_schema)
        
        results_with_feedback = await client.generate_batch(questions_with_feedback, json_format=rules_schema)
        
        os.makedirs(os.path.join(OUTPUT_PATH, DATASET_NAME), exist_ok=True)
        with open(os.path.join(OUTPUT_PATH, DATASET_NAME, "without_feedback_rules.json"), "w") as fout:
            json.dump(results_no_feedback, fout, indent=4, ensure_ascii=False)
        with open(os.path.join(OUTPUT_PATH, DATASET_NAME, "with_feedback_rules.json"), "w") as fout:
            json.dump(results_with_feedback, fout, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="quick start")
    parser.add_argument("--model", type=str, default="Qwen3-8B", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--output_root_path", type=str, default="./UNO/code/output", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--config_path", type=str, default="path/to/configs/datasets", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--dir_name", type=str, default="rules")
    parser.add_argument("--data_type", type=str, default="Long-Short", help="model name used in your deployment/model service endpoint")
    args = parser.parse_args()
    global MODEL_NAME, OUTPUT_PATH, CONFIG_PATH, DATA_TYPE
    CONFIG_PATH = os.path.join(HOME_PATH, args.config_path)
    MODEL_NAME = args.model 
    DATA_TYPE = args.data_type

    OUTPUT_PATH = os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, args.dir_name)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    asyncio.run(main())
