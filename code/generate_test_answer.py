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

from vllm_client import VllmAsyncClient, load_one_lora, unload_all_lora_adapters
import json
import os
import asyncio
from datasets import load_dataset

HOME_PATH = os.getenv("HOME_PATH", ".")
API_URL = "http://localhost:8000/v1/chat/completions"


async def main():

    client = VllmAsyncClient(
        base_model_name=MODEL_NAME,
    )

    with open(os.path.join(BEST_TEST_CONFIG_ABS_PATH), "r") as fin:
        dataset = json.load(fin)

    all_ckpt_set = set()
    for item in dataset:
        all_ckpt_set.add(item["ckpt"])
    all_ckpt_set = list(all_ckpt_set)
    ckpt_to_name = {}
    for i, ckpt in enumerate(all_ckpt_set):
        if ckpt != "init":
            assert load_one_lora(lora_path_dir=ckpt, lora_name=f"lora_{i}"), f"Failed to load lora from {ckpt}"
        ckpt_to_name[ckpt] = f"lora_{i}" if ckpt != "init" else None
    ckpt_to_items = {}
    results = []
    for item in dataset:
        ckpt = item["ckpt"]
        if ckpt not in ckpt_to_items:
            ckpt_to_items[ckpt] = {}
        max_tokens = item["max_tokens"]
        if max_tokens not in ckpt_to_items[ckpt]:
            ckpt_to_items[ckpt][max_tokens] = []
        ckpt_to_items[ckpt][max_tokens].append(item)

    for ckpt in ckpt_to_items:
        lora_name = ckpt_to_name[ckpt]
        for max_tokens in ckpt_to_items[ckpt]:
            items = ckpt_to_items[ckpt][max_tokens]
            responses = await client.generate_batch(questions_data=items, lora_name=lora_name, generation_config_overrides={"max_tokens": max_tokens, "temperature": 0.1})
            results.extend(responses)

    unload_all_lora_adapters([it for it in ckpt_to_name.values() if it is not None])
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="quick start")
    parser.add_argument("--model", type=str, default="Qwen3-8B", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--output_file_abs_path", type=str, default="output/generated_test_answers.json", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--best_test_config_abs_path", type=str, help="model name used in your deployment/model service endpoint")

    args = parser.parse_args()
    global MODEL_NAME, OUTPUT_PATH, BEST_TEST_CONFIG_ABS_PATH
    BEST_TEST_CONFIG_ABS_PATH = args.best_test_config_abs_path
    MODEL_NAME = args.model
    OUTPUT_PATH = args.output_file_abs_path

    asyncio.run(main())
