from cluster import QuestionClusterer
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file_name", type=str, default="train_data", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--output_root_path", type=str, default="./UNO/code/output2", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--num_clusters", type=int, default=3, help="Number of clusters to form.")
    parser.add_argument("--train_output_dir", type=str, default="train", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--config_path", type=str, default="path/to/configs/datasets", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--cluster_dir_name", type=str, default="cluster")
    parser.add_argument("--rules_dir_name", type=str, default="rules")
    parser.add_argument("--val_dir_name", type=str, default="valid")
    parser.add_argument("--epsilon", type=str, default="0.50", help="minimum win rate improvement to accept a checkpoint over init")
    parser.add_argument("--revised_answers_dir_name", type=str, default="revised_answers")
    parser.add_argument("--revised_answers_generate_num", type=int, default=3, help="Number of revised answers to generate for each question.")
    parser.add_argument("--revised_answers_key", type=str, default="new_a")
    parser.add_argument("--revise_type", type=str, default="4", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--sft_weight", type=str, default="0.99", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--sigmoid_weight", type=str, default="0.01", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--beta", type=str, default="0.1", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--lr", type=str, default="5e-4", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--rule_mode", type=str, default="w", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--judge_mode", type=str, default="all", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--judge_generate_num", type=int, default=5, help="number of suggestions to generate")
    parser.add_argument("--data_type", type=str, default="Long-Short", help="model name used in your deployment/model service endpoint")
    
    args = parser.parse_args()
    CONFIG_PATH = os.path.join(HOME_PATH, args.config_path)
    DATA_TYPE = args.data_type
    OUTPUT_PATH = os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, args.cluster_dir_name)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    with open(os.path.join(CONFIG_PATH, f"task.json"), "r") as fin:
        dataset_config = json.load(fin)
    if DATA_TYPE not in dataset_config:
        with open(os.path.join(CONFIG_PATH, f"domain.json"), "r") as fin:
            dataset_config = json.load(fin)
            
    with open(os.path.join(OUTPUT_PATH, "args.json"), "w", encoding="utf-8") as fout:
        json.dump(vars(args), fout, ensure_ascii=False, indent=2)

    idx_to_question = {}
    all_questions = []
    test_set_qa = []
    max_tokens_dict = {}
    
    cluster_result = {}
    
    for item in dataset_config[DATA_TYPE]:
        DATASET_NAME = item["dataset_name"]
        if "Locomo" in DATASET_NAME or "DialSim" in DATASET_NAME:
            continue
        cluster_result[DATASET_NAME] = []
        dataset = load_dataset("path/to/MemoryBench", DATASET_NAME)
        train_dataset = dataset.map(convert_str_to_obj)["train"]
        max_tokens_dict[DATASET_NAME] = item.get("max_tokens", 2048)
        max_tokens_dict[DATASET_NAME] = max(2048, max_tokens_dict[DATASET_NAME])
        
        for data in train_dataset:
            idx_to_question[f"{DATASET_NAME}_{data['test_idx']}"] = data["input_prompt"] if "input_prompt" in data else data["input_chat_messages"]
        
        qa_pairs = get_all_qa(train_dataset)
        for q, a, lang, idx in qa_pairs:
            question_id = f"{DATASET_NAME}_{idx}"
            all_questions.append({
                "test_idx": question_id,
                "question": q
            })
            cluster_result[DATASET_NAME].append({
                "id": question_id,
                "text": q
            })
            
        test_dataset = dataset.map(convert_str_to_obj)["test"]
        for q in test_dataset:
            question_id = f"{DATASET_NAME}_{q['test_idx']}"
            test_set_qa.append({
                "dataset_name": DATASET_NAME,
                "id": question_id,
                "prompt": q["input_prompt"] if "input_prompt" in q else q["input_chat_messages"],
                "max_tokens": max_tokens_dict[DATASET_NAME],
                "q": q["input_prompt"] if "input_prompt" in q else "\n\n".join([msg["content"] for msg in q["input_chat_messages"]]),
            })
            
    with open(os.path.join(OUTPUT_PATH, "cluster_results.json"), "w", encoding="utf-8") as fout:
        json.dump(cluster_result, fout, ensure_ascii=False, indent=2)
    
    revised_answers_path = os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, args.revised_answers_dir_name)
    files = os.listdir(revised_answers_path)
    train_data = {}
    for file in files:
        with open(os.path.join(revised_answers_path, file, f"{args.rule_mode}_{args.judge_mode}_judged_{args.revised_answers_key}_answers.json"), "r", encoding="utf-8") as fin:
            data = json.load(fin)
            for d in data:
                try:
                    if args.judge_mode == "all":
                        scores = [k.get("score", 0) for k in d["judge"]]
                    else:
                        scores = [np.mean([k.get("score", 0) for k in entry]) for entry in d["judge"]]
                    score = scores[0]
                except:
                    score = 0
                
                id_ = f"{file}_{d['test_idx']}"
                if not id_ in train_data:
                    train_data[id_] = {
                        "prompt": idx_to_question[id_],
                        "max_tokens": max_tokens_dict[file],
                        "old_a": d["old_a"],
                        "new_a": d["llm"],
                        "score": score
                    }
                elif train_data[id_]["score"] < score:
                    train_data[id_]["score"] = score
                    train_data[id_]["new_a"] = d["llm"]

    train_file_path = os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, f"{args.train_data_file_name}.json")
    with open(train_file_path, "w", encoding="utf-8") as fout:
        json.dump(train_data, fout, ensure_ascii=False, indent=2)

    print(f"Training data from {train_file_path}")
    
    cluster_result = json.load(open(os.path.join(OUTPUT_PATH, "cluster_results.json"), "r", encoding="utf-8"))
    
    cluster_to_best_ckpt = {
        cluster_key: "init" for cluster_key in cluster_result.keys()
    }

    cluster_to_best_epoch = {
        cluster_key: "init" for cluster_key in cluster_result.keys()
    }
    
    cluster_to_win_rate = {
        cluster_key: dict() for cluster_key in cluster_result.keys()
    }
    
    sigmoid_weight = args.sigmoid_weight
    sft_weight = args.sft_weight

    for cluster_key in cluster_result.keys():
        print(f"Start training DPO SFT for cluster {cluster_key}...")
        if not os.path.exists(os.path.join(OUTPUT_PATH, args.train_output_dir, f'{cluster_key}_Qwen3-8B_sigmoid{sigmoid_weight}_sft{sft_weight}', "adapter_model.safetensors")):
            subprocess.run(f"bash train/dpo_sft.sh {train_file_path} '{cluster_key}' {os.path.join(OUTPUT_PATH, 'cluster_results.json')} \
                {sigmoid_weight} {sft_weight} 4 {HOME_PATH} {LLM_PATH} {os.path.join(OUTPUT_PATH, args.train_output_dir)} {args.beta} {args.lr}", shell=True, check=True)
            sleep(20)

        lrpt = os.path.join(OUTPUT_PATH, args.train_output_dir, f'{cluster_key}_Qwen3-8B_sigmoid{sigmoid_weight}_sft{sft_weight}')
        print(f"Start generating validation answers for cluster {cluster_key}...")
        subprocess.run(f"python generate_val_answer.py --output_root_path {args.output_root_path} --data_type {args.data_type} \
            --output_dir_name {args.val_dir_name} --cluster_path {os.path.join(OUTPUT_PATH, 'cluster_results.json')} --cluster_key '{cluster_key}' \
            --config_path {os.path.join(CONFIG_PATH, 'each.json')} \
            --loras_path '{lrpt}'", shell=True, check=True)
        print(f"Start judging validation answers for cluster {cluster_key}...")

        val_output_abs_path = os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, args.val_dir_name)
        dataset_names_ = os.listdir(val_output_abs_path)
        check_point_val_results = {}
        all_ckpt_names = ["init"]
        for dataset_name_ in dataset_names_:
            if os.path.exists(os.path.join(val_output_abs_path, dataset_name_, f"{cluster_key}_answers.json")):
                if len(all_ckpt_names) == 1:
                    with open(os.path.join(val_output_abs_path, dataset_name_, f"{cluster_key}_answers.json"), "r", encoding="utf-8") as fin:
                        data = json.load(fin)
                        d = data[0]
                        all_ckpt_names.extend([key for key in d.keys() if key.startswith("checkpoint")])
                for ckpt_name in all_ckpt_names:
                    subprocess.run(f"python judge.py --output_root_path {args.output_root_path} --data_type {args.data_type} \
                        --answer_dir_name {args.val_dir_name} --answer_key {ckpt_name} --answer_file_name '{cluster_key}_answers.json' --rule_mode {args.rule_mode} --judge_mode {args.judge_mode} --generate_num {args.judge_generate_num} \
                        --rules_dir_name {args.rules_dir_name}", shell=True, check=True)
                for ckpt_name in all_ckpt_names:
                    with open(os.path.join(val_output_abs_path, dataset_name_, f"{args.rule_mode}_{args.judge_mode}_judged_{ckpt_name}_{cluster_key}_answers.json"), "r", encoding="utf-8") as fin:
                        data = json.load(fin)
                        for d in data:
                            id_ = f"{dataset_name_}_{d['test_idx']}"
                            if d["llm"].startswith("error"):
                                if id_ not in check_point_val_results:
                                    check_point_val_results[id_] = {}
                                check_point_val_results[id_][ckpt_name] = {
                                    "len": 0,
                                    "score": 0
                                }
                                continue
                            try:
                                total_judge_score = 0
                                total_judge_num = 0
                                for judge_score in d["judge"]:
                                    if args.judge_mode == "all":
                                        score = judge_score.get("score", 0)
                                    else:
                                        score = np.mean([entry.get("score", 0) for entry in judge_score])
                                    total_judge_score += score
                                    total_judge_num += 1
                                score = total_judge_score / total_judge_num if total_judge_num > 0 else 0
                                if id_ not in check_point_val_results:
                                    check_point_val_results[id_] = {}
                                check_point_val_results[id_][ckpt_name] = {
                                    "len": len(d["llm"].replace("<think>", "").replace("</think>", "").strip()),
                                    "score": score
                                }
                            except:
                                if id_ not in check_point_val_results:
                                    check_point_val_results[id_] = {}
                                check_point_val_results[id_][ckpt_name] = {
                                    "len": len(d["llm"].replace("<think>", "").replace("</think>", "").strip()),
                                    "score": 0
                                }
        ckpt_win_counts = {ckpt_name: 0 for ckpt_name in all_ckpt_names if ckpt_name != "init"}
        sorted_ckpt_names = sorted([ckpt_name for ckpt_name in all_ckpt_names if ckpt_name != "init"], key=lambda x: int(x.split("-")[-1]))
        total_counts = 0
        for id_, scores in check_point_val_results.items():
            init_score = scores.get("init", {"score": 0})["score"]
            init_len = scores.get("init", {"len": 0})["len"]
            for ckpt_name, scoredict in scores.items():
                if ckpt_name == "init":
                    continue
                if scoredict["len"] < 0.1 * init_len:
                    continue

                score = scoredict["score"]
                if score > init_score:
                    ckpt_win_counts[ckpt_name] += 1
                elif score == init_score:
                    ckpt_win_counts[ckpt_name] += 0.5
            total_counts += 1
        best_ckpt = "init"
        best_win_rate = 0.0
        for ckpt_name, win_count in ckpt_win_counts.items():
            win_rate = win_count / total_counts if total_counts > 0 else 0.0
            cluster_to_win_rate[cluster_key][ckpt_name] = win_rate
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_ckpt = ckpt_name
        cluster_to_best_ckpt[cluster_key] = best_ckpt
        cluster_to_best_epoch[cluster_key] = sorted_ckpt_names.index(best_ckpt)
        print(f"Best checkpoint for cluster {cluster_key} is {best_ckpt} with win rate {best_win_rate:.4f}")
        if best_win_rate <= float(args.epsilon):
            print("Warning: Best checkpoint does not exceed epsilon threshold, reverting to init.")
            cluster_to_best_ckpt[cluster_key] = "init"
            cluster_to_best_epoch[cluster_key] = "init"
            
    with open(os.path.join(OUTPUT_PATH, f"cluster_to_win_rate_judgenum{args.judge_generate_num}_sigmoid{sigmoid_weight}_sft{sft_weight}_eps{args.epsilon}.json"), "w", encoding="utf-8") as fout:
        json.dump(cluster_to_win_rate, fout, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_PATH, f"cluster_to_best_epoch_judgenum{args.judge_generate_num}_sigmoid{sigmoid_weight}_sft{sft_weight}_eps{args.epsilon}.json"), "w", encoding="utf-8") as fout:
        json.dump(cluster_to_best_epoch, fout, ensure_ascii=False, indent=2)
