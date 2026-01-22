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
from prompts import judge_json_schema, judge_json_list_schema, VALID_JUDGE_PROMPT_TEMPLATE, VALID_JUDGE_PER_RULES_PROMPT_TEMPLATE

HOME_PATH = os.getenv("HOME_PATH", ".")
API_URL = "http://localhost:8000/v1/chat/completions"


async def main():

    client = VllmAsyncClient(
        base_model_name=MODEL_NAME,
    )

    dataset_names = os.listdir(INPUT_PATH)

    for DATASET_NAME in dataset_names:

        input_dataset_path = os.path.join(INPUT_PATH, DATASET_NAME, ANSWER_FILE_NAME)

        if not os.path.exists(input_dataset_path):
            continue

        with open(input_dataset_path, "r", encoding="utf-8") as fin:
            predicts = json.load(fin)

        w_feedback_idx_to_rules = {}
        wo_feedback_idx_to_rules = {}
        w_feedback_rules_file_path = os.path.join(RULES_PATH, DATASET_NAME, "with_feedback_rules.json")
        wo_feedback_rules_file_path = os.path.join(RULES_PATH, DATASET_NAME, "without_feedback_rules.json")
        with open(w_feedback_rules_file_path, "r", encoding="utf-8") as f:
            w_feedback_rules = json.load(f)
        with open(wo_feedback_rules_file_path, "r", encoding="utf-8") as f:
            wo_feedback_rules = json.load(f)
        for item in w_feedback_rules:
            try:
                w_feedback_idx_to_rules[item["test_idx"]] = item["_raw_response"].get("rules", [])
            except:
                print(item)
        for item in wo_feedback_rules:
            try:
                wo_feedback_idx_to_rules[item["test_idx"]] = item["_raw_response"].get("rules", [])
            except:
                print(item)
                wo_feedback_idx_to_rules[item["test_idx"]] = []
        questions = []
        current_request_id = 0
        for item in predicts:
            test_idx = item["test_idx"]
            rules = []
            if RULE_MODE == "w":
                rules = w_feedback_idx_to_rules[test_idx] if test_idx in w_feedback_idx_to_rules else (wo_feedback_idx_to_rules[test_idx] if test_idx in wo_feedback_idx_to_rules else [])
            elif RULE_MODE == "wo":
                rules = wo_feedback_idx_to_rules[test_idx]
            else:
                rules = wo_feedback_idx_to_rules[test_idx]
                if test_idx in w_feedback_idx_to_rules:
                    rules += w_feedback_idx_to_rules[test_idx]
            if not rules:
                rules = ["The LLM response must be relevant to the question."]

            llm_responses = item[ANSWER_KEY]
            if not isinstance(llm_responses, list):
                llm_responses = [llm_responses]
            question = item["q"]
            rules_str = "\n".join([f"{i+1}. {r}" for i, r in enumerate(rules)])

            for llm_response in llm_responses:
                prompt = PROMPT_FOR_JUDGE.format(
                    question=question,
                    suggestion=rules_str,
                    answer=llm_response)
                prompt_messages = [{"role": "user", "content": prompt}]

                questions.append({
                    "test_idx": test_idx,
                    "is_w_feedback": test_idx in w_feedback_idx_to_rules,
                    "prompt": prompt_messages,
                    "old_a": item.get("old_a"),
                    "llm_response": llm_response,
                    "__unique_request_id__": current_request_id
                })
                current_request_id += 1

        question_lookup = {q["__unique_request_id__"]: q for q in questions}
        final_results_map = {}

        for j in range(GENERATE_NUM):

            generation_config = {}
            if j == 0:
                generation_config = {
                    "max_tokens": 4096,
                    "temperature": 0.0
                }
            else:
                generation_config = {
                    "max_tokens": 4096,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "seed": j + 42
                }

            results = await client.generate_batch(questions,
                                                generation_config_overrides=generation_config,
                                                json_format=SCHEMA)

            for res in results:
                req_id = res["__unique_request_id__"]

                if j == 0:
                    original_question = question_lookup[req_id]

                    final_results_map[req_id] = {
                        "test_idx": original_question["test_idx"],
                        "is_w_feedback": original_question["is_w_feedback"],
                        "old_a": original_question["old_a"],
                        "llm": original_question["llm_response"],
                        "judge": [res["_raw_response"]]
                    }
                else:
                    final_results_map[req_id]["judge"].append(res["_raw_response"])

        new_results = []
        for q in questions:
            req_id = q["__unique_request_id__"]
            new_results.append(final_results_map[req_id])

        with open(os.path.join(INPUT_PATH, DATASET_NAME, f"{RULE_MODE}_{JUDGE_MODE}_judged_{ANSWER_KEY}_{ANSWER_FILE_NAME}"), "w", encoding="utf-8") as fout:
            json.dump(new_results, fout, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="quick start")
    parser.add_argument("--model", type=str, default="Qwen3-8B", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--rule_mode", type=str, default="all", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--judge_mode", type=str, default="all", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--data_type", type=str, default="Long-Short", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--generate_num", type=int, default=5, help="number of suggestions to generate")

    parser.add_argument("--rules_dir_name", type=str, default="rules")
    parser.add_argument("--answer_dir_name", type=str, default="revised_answers")
    parser.add_argument("--answer_file_name", type=str, default="answers.json")
    parser.add_argument("--answer_key", type=str, default="new_a")

    parser.add_argument("--output_root_path", type=str, default="./UNO/code/output", help="model name used in your deployment/model service endpoint")

    parser.add_argument("--config_path", type=str, default="path/to/configs/datasets/each.json", help="model name used in your deployment/model service endpoint")
    args = parser.parse_args()
    global MODEL_NAME, CONFIG_PATH, RULE_MODE, JUDGE_MODE, SCHEMA, PROMPT_FOR_JUDGE
    global DATA_TYPE, RULES_PATH, GENERATE_NUM, INPUT_PATH, ANSWER_KEY, ANSWER_FILE_NAME
    ANSWER_FILE_NAME = args.answer_file_name
    assert ANSWER_FILE_NAME.endswith(".json"), "Answer file name must be a json file."
    ANSWER_KEY = args.answer_key

    DATA_TYPE = args.data_type
    GENERATE_NUM = args.generate_num

    RULES_PATH = os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, args.rules_dir_name)

    RULE_MODE = args.rule_mode
    JUDGE_MODE = args.judge_mode
    CONFIG_PATH = os.path.join(HOME_PATH, args.config_path)
    INPUT_PATH = os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, args.answer_dir_name)

    MODEL_NAME = args.model

    if JUDGE_MODE == "all":
        SCHEMA = judge_json_schema
        PROMPT_FOR_JUDGE = VALID_JUDGE_PROMPT_TEMPLATE
    else:
        SCHEMA = judge_json_list_schema
        PROMPT_FOR_JUDGE = VALID_JUDGE_PER_RULES_PROMPT_TEMPLATE

    asyncio.run(main())
