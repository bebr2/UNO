from datasets import Dataset
from typing import Dict
import pyarrow as pa
import json

from utils import get_dpo_data_with_valid
import datasets
import ast
def convert_str_to_obj(example):
    for col in example.keys():
        if col.startswith("dialog") or col.startswith("implicit_feedback") or col in ["input_chat_messages", "info"]:
            try:
                example[col] = ast.literal_eval(example[col])
            except (ValueError, SyntaxError):
                example[col] = json.loads(example[col])
    return example




def load_data(name, split="train"):
    dataset = datasets.load_dataset("path/to/MemoryBench", name)
    dataset = dataset.map(convert_str_to_obj)
    return dataset[split]


class DPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        cluster_path, cluster_key, data_path, valid=False
    ):
    
        if not valid:
            data, _ = get_dpo_data_with_valid(cluster_path, cluster_key, data_path)
        else:
            _, data = get_dpo_data_with_valid(cluster_path, cluster_key, data_path)
        
        super().__init__(arrow_table=pa.Table.from_pydict(data))
        
        print(data["prompt"][0], data["chosen"][0], data["rejected"][0])

    def __len__(self):
        return len(self.data)
