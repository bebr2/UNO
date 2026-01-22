import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Union
import math
import os
import json

LLM_PATH = os.getenv("LLM_PATH", "../LLM")


class RedundancyDetector:
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 8192):
        """
        初始化重排/去重模型检测器
        """
        self.device = device
        self.max_length = max_length
        print(f"Loading model: {model_path} ...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16
        ).to(self.device).eval()
        
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        print("Model loaded.")

    def _format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction, query=query, doc=doc)
        return output

    def _process_inputs(self, pairs: List[str]):
        """
        处理输入文本，添加特殊 token 并进行 padding
        """
        inputs = self.tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
            
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
            
        return inputs

    @torch.no_grad()
    def _compute_logits(self, inputs):
        """
        计算 Yes/No 的概率
        """
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def predict(self, queries: Union[str, List[str]], documents: Union[str, List[str]], batch_size: int = 1):
        """
        对外的主推理函数。
        Task 定义为：判断 Document (历史建议) 是否在语义上覆盖了 Query (新建议)。
        返回: List of dict, 包含 redundancy_score 和 novelty_score
        """
        if isinstance(queries, str): queries = [queries]
        if isinstance(documents, str): documents = [documents]
        
        task = 'Given a list of existing response suggestions in the Document, determine whether the response suggestion in the Query is already semantically covered by them.'
        all_results = []
        total = len(queries)
        
        for i in range(0, total, batch_size):
            batch_queries = queries[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            
            pairs = [self._format_instruction(task, q, d) for q, d in zip(batch_queries, batch_docs)]
            
            inputs = self._process_inputs(pairs)
            scores = self._compute_logits(inputs)
            
            for score in scores:
                redundancy_score = score
                novelty_score = 1.0 - redundancy_score
                all_results.append({
                    "redundancy_score": redundancy_score,
                    "novelty_score": novelty_score
                })
                
        return all_results
    
if __name__ == "__main__":
    MODEL_PATH = os.path.join(LLM_PATH, "Qwen3-Reranker-0.6B")
    detector = RedundancyDetector(model_path=MODEL_PATH, device="cuda:1", max_length=8192)
    import sys
    import os
    HOME_PATH = os.getenv("HOME_PATH", ".")
    from tqdm import tqdm
    root_path = sys.argv[2]
    
    task_type = sys.argv[1]

    root_path = os.path.join(HOME_PATH, root_path, task_type)

    dataset_names = os.listdir(os.path.join(root_path, "rules"))

    scores = {}
    scores_details = {}

    for dataset_name in dataset_names:
        print(f"Dataset {dataset_name} started")
        dataset_path = os.path.join(root_path, "rules", dataset_name)
        with_feedback_rules = json.load(open(os.path.join(dataset_path, "with_feedback_rules.json"), "r"))

        without_feedback_rules = json.load(open(os.path.join(dataset_path, "without_feedback_rules.json"), "r"))

        common_rules = set([r["test_idx"] for r in with_feedback_rules]) & set([r["test_idx"] for r in without_feedback_rules])

        common_rules_info = {}
        for r in with_feedback_rules:
            if r["test_idx"] in common_rules:
                try:
                    if not r["_raw_response"]["rules"]:
                        continue
                    common_rules_info[r["test_idx"]] = {
                        "with_feedback_rules": r["_raw_response"]["rules"]
                    }
                except:
                    continue

        for r in without_feedback_rules:
            if r["test_idx"] in common_rules:
                try:
                    if not r["_raw_response"]["rules"]:
                        continue
                    common_rules_info[r["test_idx"]]["without_feedback_rules"] = r["_raw_response"]["rules"]
                except:
                    continue

        common_rules_info = {k: v for k, v in common_rules_info.items() if "with_feedback_rules" in v and "without_feedback_rules" in v}

        all_results = {}
        batch_size = 16
        for i in tqdm(range(0, len(common_rules_info), batch_size)):
            batch = list(common_rules_info.items())[i : i + batch_size]
            batch_queries = []
            batch_docs = []
            test_idxes = []
            for test_idx, info in batch:
                with_rules = info["with_feedback_rules"]
                without_rules = info["without_feedback_rules"]
                for rule in with_rules:
                    batch_queries.append(rule)
                    batch_docs.append("\n".join(without_rules))
                    test_idxes.append(test_idx)

            results = detector.predict(batch_queries, batch_docs, batch_size=batch_size)
            for result, test_idx in zip(results, test_idxes):
                if test_idx not in all_results:
                    all_results[test_idx] = []
                all_results[test_idx].append(result)

        for test_idx in all_results:
            scores_details[f"{dataset_name}_{test_idx}"] = all_results[test_idx]

    with open(os.path.join(root_path, "cluster", "cluster_results.json"), "r") as f:
        cluster_results = json.load(f)
    
    for cluster_key in cluster_results:
        cluster = cluster_results[cluster_key]
        avg_novelty_score = 0.0
        min_novelty_score = 0.0
        max_novelty_score = 0.0
        scores[cluster_key] = {}
        total_count = 0
        for item in cluster:
            id_ = item["id"]
            if id_ not in scores_details:
                continue
            detail = scores_details[id_]
            avg_novelty_score += sum([r["novelty_score"] for r in detail]) / len(detail)
            min_novelty_score += min([r["novelty_score"] for r in detail])
            max_novelty_score += max([r["novelty_score"] for r in detail])
            total_count += 1
        if total_count == 0:
            total_count = 1
        scores[cluster_key]["avg_novelty_score"] = avg_novelty_score / total_count
        scores[cluster_key]["min_novelty_score"] = min_novelty_score / total_count
        scores[cluster_key]["max_novelty_score"] = max_novelty_score / total_count

    with open(os.path.join(root_path, "cluster", f"common_rules_info_scores.json"), "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
