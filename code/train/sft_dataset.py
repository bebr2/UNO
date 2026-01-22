from datasets import Dataset
from typing import Dict
import pyarrow as pa
import json

import datasets
import ast
import random
import os

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        cluster_path, cluster_key, data_path, valid=False, seed=42
    ):
        data_ = json.load(open(cluster_path, "r", encoding="utf-8"))[cluster_key]
        data_ = sorted(data_, key=lambda x: x["id"])
        random.seed(seed)
        random.shuffle(data_)
        if not valid:
            data_key = data_[:int(len(data_) * 0.8)]
        else:
            data_key = data_[int(len(data_) * 0.8):]
        
        dataset_names = os.listdir(data_path)
        
        raw_data = {}
        for name in dataset_names:
            rule_data = json.load(open(os.path.join(data_path, name, "sft_rules.json"), "r", encoding="utf-8"))
            for item in rule_data:
                raw_data[f"{name}_{item['test_idx']}"] = item

        print(f"Cluster Len: {len(data_key)}")
        
        new_data = {"prompt": [], "completion": []}

        for key in data_key:
            if key["id"] not in raw_data:
                continue
            item = raw_data[key["id"]]
            if item["answer"].strip() == "":
                continue
            new_data["prompt"].append([{
                "role": "user",
                "content": item["prompt"]
            }])
            new_data["completion"].append([{"role": "assistant", "content": item["answer"].strip()}])
        
        print(f"Data Len: {len(new_data['prompt'])}")
        super().__init__(arrow_table=pa.Table.from_pydict(new_data))
        
        # print(data["prompt"][0], data["chosen"][0], data["rejected"][0])

    def __len__(self):
        return len(self.data)