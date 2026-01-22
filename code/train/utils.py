import json
import os


import random

def sample_dataset(dataset, k=2000, seed=42):
    rng = random.Random(seed)
    return rng.sample(dataset, k)

def get_sft_data_valid_and_train(path, idx_to_question):
    data = json.load(open(path, "r", encoding="utf-8"))
    results = []
    for item in data:
        if "error" in item["answer"]:
            continue
        if not item["new_a"].strip():
            continue
        question = idx_to_question[item["test_idx"]]
        if type(question) == list:
            msg = question.copy()
        else:
            msg = [
                {"role": "user", "content": question},
            ]
        msg += [
            {"role": "assistant", "content": item["new_a"].strip()}
        ]
        results.append(msg)
    valid_num = int(len(results) * 0.1) + 1
    valid_data_idx = sample_dataset(list(range(len(results))), k=valid_num, seed=42)
    valid_data = [results[i] for i in valid_data_idx]
    train_data = [results[i] for i in range(len(results)) if i not in valid_data_idx]
    return valid_data, train_data

def get_sft_data(path, idx_to_question):
    data = json.load(open(path, "r", encoding="utf-8"))
    results = []
    for item in data:
        if "error" in item["answer"]:
            continue
        if not item["new_a"].strip():
            continue
        question = idx_to_question[item["test_idx"]]
        if type(question) == list:
            msg = question.copy()
        else:
            msg = [
                {"role": "user", "content": question},
            ]
        msg += [
            {"role": "assistant", "content": item["new_a"].strip()}
        ]
        results.append(msg)
    return results

def get_dpo_data_with_valid(cluster_path, cluster_key, data_path, seed=42):
    data_ = json.load(open(cluster_path, "r", encoding="utf-8"))[cluster_key]
    data_ = sorted(data_, key=lambda x: x["id"])
    random.seed(seed)
    random.shuffle(data_)
    data = data_[:int(len(data_) * 0.8)]
    dataset = json.load(open(data_path, "r", encoding="utf-8"))
    results = {"prompt": [], "chosen": [], "rejected": []}
    for it in data:
        test_idx = it["id"]
        if test_idx in dataset:
            item = dataset[test_idx]
            if not item["new_a"].strip():
                continue
            if item["new_a"].startswith("error"):
                continue
            if item["old_a"].startswith("error"):
                continue
            question = item["prompt"]
            if type(question) == list:
                msg = question.copy()
            else:
                msg = [
                    {"role": "user", "content": question},
                ]
            results["prompt"].append(msg)
            results["chosen"].append([{"role": "assistant", "content": item["new_a"].strip()}])
            results["rejected"].append([{"role": "assistant", "content": item["old_a"].strip()}])

    valid_data = data_[int(len(data_) * 0.8):]
    valid_results = {"prompt": [], "chosen": [], "rejected": []}
    for it in valid_data:
        test_idx = it["id"]
        if test_idx in dataset:
            item = dataset[test_idx]
            if not item["new_a"].strip():
                continue
            if item["new_a"].startswith("error"):
                continue
            if item["old_a"].startswith("error"):
                continue
            question = item["prompt"]
            if type(question) == list:
                msg = question.copy()
            else:
                msg = [
                    {"role": "user", "content": question},
                ]
            valid_results["prompt"].append(msg)
            valid_results["chosen"].append([{"role": "assistant", "content": item["new_a"].strip()}])
            valid_results["rejected"].append([{"role": "assistant", "content": item["old_a"].strip()}])
    return results, valid_results

def get_dpo_data(cluster_path, cluster_key, data_path, seed=42):
    data = json.load(open(cluster_path, "r", encoding="utf-8"))[cluster_key]
    data = sorted(data, key=lambda x: x["id"])
    random.seed(seed)
    random.shuffle(data)
    data = data[:int(len(data) * 0.8)]
    dataset = json.load(open(data_path, "r", encoding="utf-8"))
    results = {"prompt": [], "chosen": [], "rejected": []}
    for it in data:
        test_idx = it["id"]
        if test_idx in dataset:
            item = dataset[test_idx]
            if not item["new_a"].strip():
                continue
            if item["new_a"].startswith("error"):
                continue
            if item["old_a"].startswith("error"):
                continue
            question = item["prompt"]
            if type(question) == list:
                msg = question.copy()
            else:
                msg = [
                    {"role": "user", "content": question},
                ]
            results["prompt"].append(msg)
            results["chosen"].append([{"role": "assistant", "content": item["new_a"].strip()}])
            results["rejected"].append([{"role": "assistant", "content": item["old_a"].strip()}])
    return results


def qwen3_instruct_preprocess(msg, tokenizer, max_len, ignore_index=-100):
    ids = tokenizer.apply_chat_template(
        msg,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False
    )
    start = 0
    for i in range(len(ids) - 1, -1, -1):
        if ids[i-3:i+1] == [151667, 271, 151668, 271]:
            start = i + 1
            break
    labels = [ignore_index] * start + ids[start:]

    return ids, labels