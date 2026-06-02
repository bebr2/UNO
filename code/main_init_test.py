import argparse
import ast
import json
import os
import subprocess
import sys

from datasets import load_dataset


HOME_PATH = os.getenv("HOME_PATH", ".")
MEMORYBENCH_PATH = os.getenv("MEMORYBENCH_PATH")


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


def load_dataset_config(config_path, data_type):
    with open(os.path.join(config_path, "task.json"), "r", encoding="utf-8") as fin:
        dataset_config = json.load(fin)
    if data_type not in dataset_config:
        with open(os.path.join(config_path, "domain.json"), "r", encoding="utf-8") as fin:
            dataset_config = json.load(fin)
    return dataset_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate base-model test answers used by UNO reflective inference."
    )
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument("--output_root_path", type=str, default="./UNO/code/output_init_0")
    parser.add_argument("--config_path", type=str, default=os.getenv("CONFIG_PATH", "path/to/configs/datasets"))
    parser.add_argument("--cluster_dir_name", type=str, default="cluster")
    parser.add_argument("--test_cluster_file_name", type=str, default="test_set_clustered_qa.json")
    parser.add_argument("--test_output_file_name", type=str, default="test_results.json")
    parser.add_argument("--data_type", type=str, default="Long-Long")
    parser.add_argument("--small_test", action="store_true")
    args = parser.parse_args()

    config_path = os.path.join(HOME_PATH, args.config_path)
    output_path = os.path.join(HOME_PATH, args.output_root_path, args.data_type, args.cluster_dir_name)
    os.makedirs(output_path, exist_ok=True)

    dataset_config = load_dataset_config(config_path, args.data_type)
    test_set_qa = []

    for item in dataset_config[args.data_type]:
        dataset_name = item["dataset_name"]
        if "Locomo" in dataset_name or "DialSim" in dataset_name:
            continue

        if not MEMORYBENCH_PATH:
            raise ValueError("Please set MEMORYBENCH_PATH to the local MemoryBench dataset path.")
        dataset = load_dataset(MEMORYBENCH_PATH, dataset_name)
        test_dataset = dataset.map(convert_str_to_obj)["test"]
        max_tokens = item.get("max_tokens", item.get("max_output_len", 2048))
        max_tokens = max(2048, max_tokens if max_tokens is not None else 2048)

        for row in test_dataset:
            question_id = f"{dataset_name}_{row['test_idx']}"
            prompt = row["input_prompt"] if "input_prompt" in row else row["input_chat_messages"]
            q_text = row["input_prompt"] if "input_prompt" in row else "\n\n".join(
                msg["content"] for msg in row["input_chat_messages"]
            )
            test_set_qa.append({
                "id": question_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "q": q_text,
                "cluster_key": "0",
                "ckpt": "init",
            })

    if args.small_test:
        test_set_qa = test_set_qa[:10]

    test_config_path = os.path.join(output_path, args.test_cluster_file_name)
    with open(test_config_path, "w", encoding="utf-8") as fout:
        json.dump(test_set_qa, fout, ensure_ascii=False, indent=2)

    subprocess.run(
        [
            sys.executable,
            "generate_test_answer.py",
            "--model",
            args.model,
            "--output_file_abs_path",
            os.path.join(output_path, args.test_output_file_name),
            "--best_test_config_abs_path",
            test_config_path,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
