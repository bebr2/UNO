from collections import Counter
import json
from typing import List, Dict, Any
from prompts import get_all_qa, get_suggestion_prompt
import ast
import re
import regex
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import numpy as np
import string
import os
from vllm_client import VllmAsyncClient, load_one_lora, unload_all_lora_adapters
import json
import os
import asyncio
from datasets import load_dataset

HOME_PATH = os.getenv("HOME_PATH", ".")
API_URL = "http://localhost:8000/v1/chat/completions"

import random

async def main():
    questions_dict = {}
    if MODEL_NAME == "Qwen3-8B":
        folder = "output_init_0"
    else:
        folder = f"output_{MODEL_NAME}_init_0"
    with open(f"{HOME_PATH}/UNO/code/{folder}/{DATA_TYPE}/cluster/test_results.json") as f:
        data = json.load(f)
        for item in data:
            new_item = {
                "id": item["id"],
                "prompt": f"Provide suggestions to make the answer better. Your output must be some rules, using number to order them.\n\n## User Query\n\n{item['prompt']}\n\n## Initial Answer\n\n{item['_raw_response']}",
                "old_a": item["_raw_response"],
                "max_tokens": item["max_tokens"],
                "q": item["q"]
            }
            questions_dict[item["id"]] = new_item

    if os.path.exists(os.path.join(CLUSTER_FILE_ABS_PATH, "test_set_clustered_qa.json")):
        cluster_path = os.path.join(CLUSTER_FILE_ABS_PATH, "test_set_clustered_qa.json")
    elif os.path.exists(os.path.join(CLUSTER_FILE_ABS_PATH, "test_set_clustered_qa_epoch0.json")):
        cluster_path = os.path.join(CLUSTER_FILE_ABS_PATH, "test_set_clustered_qa_epoch0.json")
    else:
        cluster_path = os.path.join(CLUSTER_FILE_ABS_PATH, "test_set_clustered_db_qa_epoch0.json")
    with open(cluster_path, "r", encoding="utf-8") as fin:
        cluster_data = json.load(fin)
        
    loras = os.listdir(LORA_PATH)
    cluster_to_lora = {k.split("_")[0]: os.path.join(LORA_PATH, k) for k in loras}
    cluster_to_ids = {}
    for item in cluster_data:
        cluster_id = item["cluster_key"]
        if cluster_id not in cluster_to_ids:
            cluster_to_ids[cluster_id] = []
        cluster_to_ids[cluster_id].append(item["id"])

    client = VllmAsyncClient(
        base_model_name=MODEL_NAME,
    )

    results = []
    failed_clusters = []
    
    for cluster_key, lora_path in cluster_to_lora.items():
        is_success_lora = load_one_lora(lora_path, cluster_key)
        if not is_success_lora:
            failed_clusters.append(cluster_key)
            continue
    success_loras = list(set(cluster_to_lora.keys()) - set(failed_clusters))
    
    for ckpt in success_loras:
        
        ids = cluster_to_ids[ckpt]
        questions_all = []
        for id_ in ids:
            item = questions_dict[id_]
            questions_all.append(item)
            
        max_token_to_questions = {}
        for item in questions_all:
            max_tk = item["max_tokens"]
            if max_tk not in max_token_to_questions:
                max_token_to_questions[max_tk] = []
            max_token_to_questions[max_tk].append(item)

        for max_tk, questions in max_token_to_questions.items():
            responses = await client.generate_batch(questions_data=questions, lora_name=ckpt, generation_config_overrides={"max_tokens": questions[0]["max_tokens"], "temperature": 0.1})
            for i in range(len(responses)):
                responses[i]["rules"] = responses[i].pop("_raw_response").replace("</think>", "").replace("<think>", "").strip()
                lang = "en"
                if re.search(r"[\u4e00-\u9fa5]", item["q"]):
                    lang = "zh"
                responses[i]["prompt"] = get_suggestion_prompt(rules=responses[i]["rules"], lang=lang, old_answer=responses[i]["old_a"], q=responses[i]["q"], version="v2")
                
            responses_new = await client.generate_batch(questions_data=responses, generation_config_overrides={"max_tokens": questions[0]["max_tokens"]})
            
            results.extend(responses_new)

            if SMALL_TEST:
                break
        if SMALL_TEST:
            break
            
    unload_all_lora_adapters(success_loras)

    for cluster_key in failed_clusters:
        results.extend([{"id": id_, "rules": "", "_raw_response": "failed"} for id_ in cluster_to_ids[cluster_key]])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="quick start")
    parser.add_argument("--model", type=str, default="Qwen3-8B", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--data_type", type=str, default="Long-Short", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--cluster_file_abs_path", type=str, default="", help="absolute path to cluster file")
    parser.add_argument("--output_file_name", type=str, default="test_with_rules.json", help="output directory name")
    parser.add_argument("--small_test", action="store_true", help="whether to run a small test")
    parser.add_argument("--lora_path", type=str, help="path to loras directory")
    args = parser.parse_args()
    global MODEL_NAME, OUTPUT_PATH, LORA_PATH, DATA_TYPE, SMALL_TEST, CLUSTER_FILE_ABS_PATH
    CLUSTER_FILE_ABS_PATH = args.cluster_file_abs_path
    SMALL_TEST = args.small_test
    MODEL_NAME = args.model 
    OUTPUT_PATH = os.path.join(CLUSTER_FILE_ABS_PATH, args.output_file_name)
    LORA_PATH = args.lora_path
    DATA_TYPE = args.data_type

    asyncio.run(main())
