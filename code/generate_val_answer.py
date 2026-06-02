import json
from collections import Counter
import json
from typing import List, Dict, Any
from prompts import get_all_qa
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

import random

def get_dpo_data(cluster_path, cluster_key, config_path, seed=42):
    data = json.load(open(cluster_path, "r", encoding="utf-8"))[cluster_key]
    data = sorted(data, key=lambda x: x["id"])
    random.seed(seed)
    random.shuffle(data)
    data = data[int(len(data) * 0.8):]
    dataset_config = json.load(open(config_path, "r", encoding="utf-8"))
    all_dataset_name = {it["id"].split("_")[0] for it in data}
    all_dataset_name_to_max_tokens = {}
    for dataset_name in all_dataset_name:
        max_tokens = max(2048, dataset_config[dataset_name].get("max_tokens", 2048))
        all_dataset_name_to_max_tokens[dataset_name] = max_tokens
    all_questions = {}
    for dataset_name in all_dataset_name:
        if not MEMORYBENCH_PATH:
            raise ValueError("Please set MEMORYBENCH_PATH to the local MemoryBench dataset path.")
        dataset = load_dataset(MEMORYBENCH_PATH, dataset_name)
        train_dataset = dataset.map(convert_str_to_obj)["train"]
        for item in train_dataset:
            question_id = f"{dataset_name}_{item['test_idx']}"
            msg = item["input_chat_messages"].copy() if "input_chat_messages" in item else \
                [{"role": "user", "content": item["input_prompt"]}]
            all_questions[question_id] = msg

    results = []
    for it in data:
        test_idx = it["id"]
        dataset_name = test_idx.split("_")[0]
        msg = all_questions[test_idx]
        results.append({
            "id": test_idx,
            "prompt": msg,
            "max_tokens": all_dataset_name_to_max_tokens[dataset_name],
        })
    return results


async def main(cluster_path, cluster_key, config_path, is_small_test=False):
    dataset = get_dpo_data(cluster_path, cluster_key, config_path)
    if is_small_test:
        dataset = dataset[:3]
    dataset_by_max_tokens = {}
    for item in dataset:
        max_tokens = item["max_tokens"]
        if max_tokens not in dataset_by_max_tokens:
            dataset_by_max_tokens[max_tokens] = []
        dataset_by_max_tokens[max_tokens].append(item)

    client = VllmAsyncClient(
        base_model_name=MODEL_NAME,
    )

    success_loras = load_all_lora_adapters(LORAS_PATH)
    results = {}
    for ckpt in ["init"] + success_loras:
        for max_tokens, items in dataset_by_max_tokens.items():
            responses = await client.generate_batch(questions_data=items, lora_name=ckpt if ckpt != "init" else None, generation_config_overrides={"max_tokens": max_tokens})
            for response in responses:
                test_idx = int(response["id"].split("_")[-1])
                dataset_name = response["id"].split("_")[0]
                if dataset_name not in results:
                    results[dataset_name] = []
                for i, res in enumerate(results[dataset_name]):
                    if res["test_idx"] == test_idx:
                        results[dataset_name][i][ckpt] = response["_raw_response"]
                        break
                else:
                    q = ""
                    for msg in response["prompt"]:
                        q += msg["content"] + "\n\n"
                    q = q.strip()
                    results[dataset_name].append({
                        "test_idx": test_idx,
                        "q": q,
                        ckpt: response["_raw_response"]
                    })

    unload_all_lora_adapters(success_loras)
    for dataset_name, res in results.items():
        os.makedirs(os.path.join(OUTPUT_PATH, dataset_name), exist_ok=True)
        with open(os.path.join(OUTPUT_PATH, dataset_name, f"{cluster_key}_answers.json"), "w", encoding="utf-8") as fout:
            json.dump(res, fout, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="quick start")
    parser.add_argument("--model", type=str, default="Qwen3-8B", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--output_root_path", type=str, default="./UNO/code/output", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--data_type", type=str, default="Long-Short", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--output_dir_name", type=str, default="valid", help="output directory name")
    parser.add_argument("--small_test", action="store_true", help="whether to use small test")
    parser.add_argument("--cluster_path", type=str, help="path to cluster json file")
    parser.add_argument("--cluster_key", type=str, help="key in cluster json file")
    parser.add_argument("--loras_path", type=str, help="path to loras directory")
    parser.add_argument("--config_path", type=str, help="path to original data json file")
    args = parser.parse_args()
    global MODEL_NAME, OUTPUT_PATH, LORAS_PATH, DATA_TYPE
    DATA_TYPE = args.data_type
    MODEL_NAME = args.model
    OUTPUT_PATH = os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, args.output_dir_name)
    LORAS_PATH = args.loras_path

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    asyncio.run(main(args.cluster_path, args.cluster_key, args.config_path, is_small_test=args.small_test))
