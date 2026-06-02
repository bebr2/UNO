import argparse
import os
import ast
import json
from datasets import load_dataset
from prompts import get_all_qa
import subprocess
import numpy as np
from time import sleep
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
import re

HOME_PATH = os.getenv("HOME_PATH", ".")
LLM_PATH = os.getenv("LLM_PATH", "../LLM")
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
    parser.add_argument("--copy_root_path", type=str)
    parser.add_argument("--small_test", action="store_true", help="Use a small test dataset for quick debugging.")
    parser.add_argument("--train_data_file_name", type=str, default="train_data", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--output_root_path", type=str, default="./UNO/code/output2", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--distance_threshold", type=str, default="4", help="Distance threshold for clustering.")
    parser.add_argument("--train_output_dir", type=str, default="train", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--config_path", type=str, default="path/to/configs/datasets", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--cluster_dir_name", type=str, default="cluster")
    parser.add_argument("--rules_dir_name", type=str, default="rules")
    parser.add_argument("--val_dir_name", type=str, default="valid")
    parser.add_argument("--epsilon", type=str, default="0.50", help="minimum win rate improvement to accept a checkpoint over init")
    parser.add_argument("--bleu_ratio", type=str, default="0.05", help="minimum bleu improvement to accept a checkpoint over init")
    parser.add_argument("--revised_answers_dir_name", type=str, default="revised_answers")
    parser.add_argument("--revised_answers_generate_num", type=int, default=3, help="Number of revised answers to generate for each question.")
    parser.add_argument("--revised_answers_key", type=str, default="new_a")
    parser.add_argument("--revise_type", type=str, default="1_4", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--test_cluster_file_name", type=str, default="test_set_clustered_qa.json", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--test_output_file_name", type=str, default="test_results.json", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--sft_weight", type=str, default="0.5", help="model name used in your deployment/model service endpoint")
    parser.add_argument("--sigmoid_weight", type=str, default="0.5", help="model name used in your deployment/model service endpoint")
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

    if args.copy_root_path:
        copy_root_path = os.path.join(HOME_PATH, args.copy_root_path, DATA_TYPE)
        subprocess.run(f"cp -r {copy_root_path}/revised_answers {os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE)}", shell=True)
        subprocess.run(f"cp -r {copy_root_path}/rules {os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE)}", shell=True)

    train_file_path = os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, f"{args.train_data_file_name}.json")

    if args.copy_root_path:
        subprocess.run(f"cp {os.path.join(HOME_PATH, args.copy_root_path, DATA_TYPE, f'{args.train_data_file_name}.json')} {train_file_path}", shell=True, check=True)
    elif not os.path.exists(train_file_path):
        subprocess.run(f"python generate_rules.py --model {args.model} --output_root_path {args.output_root_path} --data_type {args.data_type} --dir_name {args.rules_dir_name}", shell=True, check=True)
        subprocess.run(f"python revise_answer_multiple.py --model {args.model} --output_root_path {args.output_root_path} --data_type {args.data_type} --rules_dir_name {args.rules_dir_name} --output_dir_name {args.revised_answers_dir_name} --generate_num {args.revised_answers_generate_num} --revise_type {args.revise_type}", shell=True, check=True)
        subprocess.run(f"python judge.py --model {args.model} --output_root_path {args.output_root_path} --data_type {args.data_type} --answer_dir_name {args.revised_answers_dir_name} --answer_key {args.revised_answers_key} --rule_mode {args.rule_mode} --judge_mode {args.judge_mode} --generate_num 1", shell=True, check=True)

    with open(os.path.join(CONFIG_PATH, f"task.json"), "r") as fin:
        dataset_config = json.load(fin)
    if DATA_TYPE not in dataset_config:
        with open(os.path.join(CONFIG_PATH, f"domain.json"), "r") as fin:
            dataset_config = json.load(fin)

    idx_to_question = {}
    all_questions = []
    test_set_qa = []
    max_tokens_dict = {}

    for item in dataset_config[DATA_TYPE]:
        DATASET_NAME = item["dataset_name"]
        if "Locomo" in DATASET_NAME or "DialSim" in DATASET_NAME:
            continue
        if not MEMORYBENCH_PATH:
            raise ValueError("Please set MEMORYBENCH_PATH to the local MemoryBench dataset path.")
        dataset = load_dataset(MEMORYBENCH_PATH, DATASET_NAME)
        train_dataset = dataset.map(convert_str_to_obj)["train"]
        max_tokens_dict[DATASET_NAME] = item.get("max_tokens", 2048)
        max_tokens_dict[DATASET_NAME] = max(2048, max_tokens_dict[DATASET_NAME])

        for data in train_dataset:
            idx_to_question[f"{DATASET_NAME}_{data['test_idx']}"] = data["input_prompt"] if "input_prompt" in data else data["input_chat_messages"]

        with open(os.path.join(HOME_PATH, args.output_root_path, DATA_TYPE, args.rules_dir_name, DATASET_NAME, "with_feedback_rules.json"), "r", encoding="utf-8") as fin:
            rules = json.load(fin)

        w_rules_dict = {r["test_idx"]: r for r in rules}
        qa_pairs = get_all_qa(train_dataset)
        for q, a, lang, idx in qa_pairs:
            if idx not in w_rules_dict:
                continue
            if type(w_rules_dict[idx]["_raw_response"]) == str:
                continue
            question_id = f"{DATASET_NAME}_{idx}"
            all_questions.append({
                "test_idx": question_id,
                "question": q + "[Rules]\n" + "\n".join(w_rules_dict[idx]["_raw_response"]["rules"])
            })

        test_dataset = dataset.map(convert_str_to_obj)["test"]
        for q in test_dataset:
            question_id = f"{DATASET_NAME}_{q['test_idx']}"
            test_set_qa.append({
                "id": question_id,
                "prompt": q["input_prompt"] if "input_prompt" in q else q["input_chat_messages"],
                "max_tokens": max_tokens_dict[DATASET_NAME],
                "q": q["input_prompt"] if "input_prompt" in q else "\n\n".join([msg["content"] for msg in q["input_chat_messages"]]),
            })

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
                        "prompt":  idx_to_question[id_],
                        "max_tokens": max_tokens_dict[file],
                        "old_a": d["old_a"],
                        "new_a": d["llm"],
                        "score": score
                    }
                elif train_data[id_]["score"] < score:
                    train_data[id_]["score"] = score
                    train_data[id_]["new_a"] = d["llm"]

    with open(train_file_path, "w", encoding="utf-8") as fout:
        json.dump(train_data, fout, ensure_ascii=False, indent=2)

    if args.small_test:
        with open(train_file_path, "r", encoding="utf-8") as fin:
            train_data = json.load(fin)
        import random
        random.seed(42)
        selected_keys = random.sample(list(train_data.keys()), min(40, len(train_data)))
        small_train_data = {key: train_data[key] for key in selected_keys}
        with open(train_file_path, "w", encoding="utf-8") as fout:
            json.dump(small_train_data, fout, ensure_ascii=False, indent=2)
    print(f"Training data saved to {train_file_path}")

    from cluster_a import QuestionClusterer

    clusterer = QuestionClusterer(distance_threshold=float(args.distance_threshold), model_name="Qwen/Qwen3-Embedding-0.6B")

    with open(os.path.join(OUTPUT_PATH, "args.json"), "w", encoding="utf-8") as fout:
        json.dump(vars(args), fout, ensure_ascii=False, indent=2)

    clusterer.fit(all_questions)
    clusterer.print_cluster_status()
    clusterer.save_model(OUTPUT_PATH)
    clusterer.save_cluster_results_json(OUTPUT_PATH)

    with open(os.path.join(OUTPUT_PATH, "cluster_results.json"), "r") as fin:
        cluster_results = json.load(fin)

    key_to_dataset_name = {}

    for key in cluster_results.keys():
        dataset_name = [dn["id"].split("_")[0].split("-")[-1] for dn in cluster_results[key]]
        dataset_name = max(set(dataset_name), key=dataset_name.count)
        key_to_dataset_name[key] = dataset_name
        wrong_assigned = [dn for dn in cluster_results[key] if not dataset_name in dn["id"]]
        print(f"Cluster {key} assigned to dataset {dataset_name} has {len(wrong_assigned)} wrong assigned questions out of {len(cluster_results[key])}")
        with open(os.path.join(OUTPUT_PATH, "cluster_log.txt"), "a", encoding="utf-8") as fout:
            fout.write(f"Cluster {key} assigned to dataset {dataset_name} has {len(wrong_assigned)} wrong assigned questions out of {len(cluster_results[key])}\n")
        wrong_dataset_names = [dn["id"].split("_")[0] for dn in wrong_assigned]
        for name in set(wrong_dataset_names):
            print(f"Cluster {key} assigned to dataset {dataset_name} has {wrong_dataset_names.count(name)} wrong assigned questions from dataset {name}")
            with open(os.path.join(OUTPUT_PATH, "cluster_log.txt"), "a", encoding="utf-8") as fout:
                fout.write(f"Cluster {key} assigned to dataset {dataset_name} has {wrong_dataset_names.count(name)} wrong assigned questions from dataset {name}\n")

    test_cluster_keys = clusterer.predict([it["q"] for it in test_set_qa])
    wrong_count = 0
    for i, it in enumerate(test_set_qa):
        assigned_cluster = test_cluster_keys[i]
        assigned_dataset = key_to_dataset_name[assigned_cluster]
        true_dataset = it["id"].split("_")[0]
        if not assigned_dataset in true_dataset:
            wrong_count += 1
    print(f"For DATA_TYPE {DATA_TYPE}, test set wrong assigned questions: {wrong_count} out of {len(test_set_qa)}")
    with open(os.path.join(OUTPUT_PATH, "cluster_log.txt"), "a", encoding="utf-8") as fout:
        fout.write(f"For DATA_TYPE {DATA_TYPE}, test set wrong assigned questions: {wrong_count} out of {len(test_set_qa)}\n")

    subprocess.run(f"python hinfo.py {args.data_type} {args.output_root_path}", shell=True, check=True)

    cluster_result = json.load(open(os.path.join(OUTPUT_PATH, clusterer._results_file), "r", encoding="utf-8"))

    cluster_to_best_ckpt = {cluster_key: "init" for cluster_key in cluster_result.keys()}
    cluster_to_best_epoch = {cluster_key: "init" for cluster_key in cluster_result.keys()}
    cluster_to_win_rate = {cluster_key: dict() for cluster_key in cluster_result.keys()}
    cluster_to_best_ckpt_by_bleu = {cluster_key: "init" for cluster_key in cluster_result.keys()}
    cluster_to_best_epoch_by_bleu = {cluster_key: "init" for cluster_key in cluster_result.keys()}
    cluster_to_win_rate_by_bleu = {cluster_key: dict() for cluster_key in cluster_result.keys()}

    cluster_list_to_ckpt = [{
        cluster_key: "init" for cluster_key in cluster_result.keys()
    } for _ in range(8)]

    sigmoid_weight = args.sigmoid_weight
    sft_weight = args.sft_weight

    for cluster_key in cluster_result.keys():
        print(f"Start training DPO SFT for cluster {cluster_key}...")

        if not os.path.exists(os.path.join(OUTPUT_PATH, args.train_output_dir, f'{cluster_key}_{args.model}_sigmoid{sigmoid_weight}_sft{sft_weight}', "adapter_model.safetensors")):
            subprocess.run(f"bash train/dpo_sft.sh {train_file_path} {cluster_key} {os.path.join(OUTPUT_PATH, clusterer._results_file)} \
                {sigmoid_weight} {sft_weight} 4 {HOME_PATH} {LLM_PATH} {os.path.join(OUTPUT_PATH, args.train_output_dir)} {args.beta} {args.lr} {args.model}", shell=True, check=True)
            sleep(20)

            print(f"Start generating validation answers for cluster {cluster_key}...")
            if args.small_test:
                if not os.path.exists(os.path.join(OUTPUT_PATH, "../", args.val_dir_name)):
                    subprocess.run(f"python generate_val_answer.py --model {args.model} --output_root_path {args.output_root_path} --data_type {args.data_type} \
                        --output_dir_name {args.val_dir_name} --cluster_path {os.path.join(OUTPUT_PATH, clusterer._results_file)} --cluster_key {cluster_key} \
                        --config_path {os.path.join(CONFIG_PATH, 'each.json')} \
                        --loras_path {os.path.join(OUTPUT_PATH, args.train_output_dir, f'{cluster_key}_{args.model}_sigmoid{sigmoid_weight}_sft{sft_weight}')} --small_test", shell=True, check=True)
            else:
                subprocess.run(f"python generate_val_answer.py --model {args.model} --output_root_path {args.output_root_path} --data_type {args.data_type} \
                    --output_dir_name {args.val_dir_name} --cluster_path {os.path.join(OUTPUT_PATH, clusterer._results_file)} --cluster_key {cluster_key} \
                    --config_path {os.path.join(CONFIG_PATH, 'each.json')} \
                    --loras_path {os.path.join(OUTPUT_PATH, args.train_output_dir, f'{cluster_key}_{args.model}_sigmoid{sigmoid_weight}_sft{sft_weight}')}", shell=True, check=True)

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
                    if args.small_test and os.path.exists(os.path.join(OUTPUT_PATH, "../", args.val_dir_name)):
                        continue
                    subprocess.run(f"python judge.py --model {args.model} --output_root_path {args.output_root_path} --data_type {args.data_type} \
                        --answer_dir_name {args.val_dir_name} --answer_key {ckpt_name} --answer_file_name {cluster_key}_answers.json --rule_mode {args.rule_mode} --judge_mode {args.judge_mode} --generate_num {args.judge_generate_num} \
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
                                    "score": 0,
                                    "response": ""
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
                                    "score": score,
                                    "response": d["llm"].replace("<think>", "").replace("</think>", "").strip()
                                }
                            except:
                                if id_ not in check_point_val_results:
                                    check_point_val_results[id_] = {}
                                check_point_val_results[id_][ckpt_name] = {
                                    "len": len(d["llm"].replace("<think>", "").replace("</think>", "").strip()),
                                    "score": 0,
                                    "response": d["llm"].replace("<think>", "").replace("</think>", "").strip()
                                }

        ckpt_win_counts = {ckpt_name: 0 for ckpt_name in all_ckpt_names if ckpt_name != "init"}
        sorted_ckpt_names = sorted([ckpt_name for ckpt_name in all_ckpt_names if ckpt_name != "init"], key=lambda x: int(x.split("-")[-1]))

        for ii in range(8):
            cluster_list_to_ckpt[ii][cluster_key] = sorted_ckpt_names[ii]

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
        cluster_to_best_epoch[cluster_key] = sorted_ckpt_names.index(best_ckpt) if best_ckpt != "init" else "init"

        print(f"Best checkpoint for cluster {cluster_key} is {best_ckpt} with win rate {best_win_rate:.4f}")
        if best_win_rate <= float(args.epsilon):
            print("Warning: Best checkpoint does not exceed epsilon threshold, reverting to init.")
            cluster_to_best_ckpt[cluster_key] = "init"
            cluster_to_best_epoch[cluster_key] = "init"

        bleu_ckpt_scores = {ckpt_name: {} for ckpt_name in all_ckpt_names if ckpt_name != "init"}
        for id_, scores in check_point_val_results.items():
            init_response = scores.get("init", {"response": ""})["response"]
            for ckpt_name, scoredict in scores.items():
                if ckpt_name == "init":
                    continue
                response = scoredict["response"]
                bleu_score = calculate_metrics(init_response, response)
                bleu_ckpt_scores[ckpt_name][id_] = bleu_score

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
                bleu_score = bleu_ckpt_scores[ckpt_name][id_]
                if score > init_score and bleu_score > float(args.bleu_ratio):
                    ckpt_win_counts[ckpt_name] += 1
                elif score == init_score and bleu_score > float(args.bleu_ratio):
                    ckpt_win_counts[ckpt_name] += 0.5
            total_counts += 1
        best_ckpt = "init"
        best_win_rate = 0.0
        for ckpt_name, win_count in ckpt_win_counts.items():
            win_rate = win_count / total_counts if total_counts > 0 else 0.0
            cluster_to_win_rate_by_bleu[cluster_key][ckpt_name] = win_rate
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_ckpt = ckpt_name
        cluster_to_best_ckpt_by_bleu[cluster_key] = best_ckpt
        cluster_to_best_epoch_by_bleu[cluster_key] = sorted_ckpt_names.index(best_ckpt) if best_ckpt != "init" else "init"

        if best_win_rate < float(args.epsilon):
            print("Warning: Best checkpoint by BLEU does not exceed epsilon threshold, reverting to init.")
            cluster_to_best_ckpt_by_bleu[cluster_key] = "init"
            cluster_to_best_epoch_by_bleu[cluster_key] = "init"

    with open(os.path.join(OUTPUT_PATH, f"cluster_to_win_rate.json"), "w", encoding="utf-8") as fout:
        json.dump(cluster_to_win_rate, fout, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_PATH, f"cluster_to_best_epoch.json"), "w", encoding="utf-8") as fout:
        json.dump(cluster_to_best_epoch, fout, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_PATH, f"cluster_to_win_rate_by_bleu.json"), "w", encoding="utf-8") as fout:
        json.dump(cluster_to_win_rate_by_bleu, fout, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_PATH, f"cluster_to_best_epoch_by_bleu.json"), "w", encoding="utf-8") as fout:
        json.dump(cluster_to_best_epoch_by_bleu, fout, ensure_ascii=False, indent=2)

    print("Start generating test answers...")
    test_cluster_keys = clusterer.predict([it["q"] for it in test_set_qa])
    if args.small_test:
        test_cluster_keys = test_cluster_keys[:10]
        test_set_qa = test_set_qa[:10]

    for j in range(8 if not args.small_test else 1):
        best_test_set_qa = test_set_qa.copy()
        for i, it in enumerate(test_set_qa):
            best_test_set_qa[i]["cluster_key"] = test_cluster_keys[i]
            best_test_set_qa[i]["ckpt"] = cluster_list_to_ckpt[j][test_cluster_keys[i]]
            if best_test_set_qa[i]["ckpt"] != "init":
                best_test_set_qa[i]["ckpt"] = os.path.join(OUTPUT_PATH, args.train_output_dir, f"{test_set_qa[i]['cluster_key']}_{args.model}_sigmoid{sigmoid_weight}_sft{sft_weight}", cluster_list_to_ckpt[j][test_cluster_keys[i]])

        with open(os.path.join(OUTPUT_PATH, f"test_set_clustered_qa_epoch{j}.json"), "w", encoding="utf-8") as fout:
            json.dump(best_test_set_qa, fout, ensure_ascii=False, indent=2)

        subprocess.run(f"python generate_test_answer.py --model {args.model} --output_file_abs_path {os.path.join(OUTPUT_PATH, f'test_results_epoch{j}.json')} \
            --best_test_config_abs_path {os.path.join(OUTPUT_PATH, f'test_set_clustered_qa_epoch{j}.json')}", shell=True, check=True)
