import os
from typing import Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
import torch
import json
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Literal
from transformers import set_seed
import random

def set_seed_for_all(seed: int = 42):
    set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    cluster_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    cluster_key: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    filter_ratio: float = field(
        default=0.1, metadata={"help": "The ratio of data to filter out before training."}
    )
    filter_mean_threshold: float = field(
        default=0.25, metadata={"help": "The mean threshold ratio for filtering."}
    )
    filter_loss_type: str = field(
        default="sft", metadata={"help": "The loss type for filtering: 'dpo' or 'sft'."}
    )

def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, DPOConfig)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed_for_all(training_args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    model_ref = None
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.max_length,
    )
    if type(tokenizer) == bool:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.max_length,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from dpo_dataset import DPODataset
    train_dataset = DPODataset(
        cluster_path=data_args.cluster_path,
        cluster_key=data_args.cluster_key,
        data_path=data_args.data_path,
    )
    lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "v_proj",
                "o_proj",
                "k_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model()

if __name__ == "__main__":
    train()
