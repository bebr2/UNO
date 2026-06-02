# UNO: User log-driveN Optimization

UNO is a framework for improving LLM systems with user logs. It distills user feedback into semi-structured rules and preference data, clusters logs by query-rule semantics, estimates the model-log cognitive gap, and routes future requests through Primary or Reflective experience modules.

Chinese documentation is available in [README_zh.md](README_zh.md).

## Code Map

| Paper component | Code entry | Role |
| --- | --- | --- |
| Preprocessing | `generate_rules.py`, `revise_answer_multiple.py`, `judge.py`, `main.py` | Distill feedback rules, generate revised answers, score candidates, and build preference data. |
| Query-and-feedback clustering | `cluster_a.py`, `main.py` | Cluster logs with query and rule embeddings. |
| Cognitive gap assessment | `hinfo.py`, `main.py` | Use Qwen3-Reranker-0.6B to decide whether a cluster should be treated as high-gap. |
| Primary Experience Module | `main.py`, `train/dpo_sft.sh` | Train cluster-specific Expert LoRAs with DPO + SFT loss. |
| Reflective Experience Module | `main_reverse_user.py`, `get_rule_train_data.py`, `train/sft.sh` | Train Critic LoRAs that generate revision suggestions. |
| Output aggregation | `aggregate_uno_outputs.py` | Build final UNO-Single and UNO answer files from init, primary, and reflective outputs. |

## Environment

```bash
conda create -n uno python=3.11.14
conda activate uno
pip install -r requirements.txt
```

Required local resources:

- `Qwen3-8B`
- `Qwen3-Embedding-0.6B`
- `Qwen3-Reranker-0.6B`
- `qwen3_nonthinking.jinja`
- MemoryBench dataset
- MemoryBench `configs/datasets`

## Paths

Set these variables before running:

```bash
export HOME_PATH=/path/to/parent/of/UNO
export LLM_PATH=/path/to/model_root
export MEMORYBENCH_PATH=/path/to/MemoryBench
export CONFIG_PATH=/path/to/MemoryBench/configs/datasets
```

`HOME_PATH` must be the parent directory of this repository. If the repo is `/data/code/UNO`, set `HOME_PATH=/data/code`.

Expected model layout:

```text
$LLM_PATH/
  Qwen3-8B/
    qwen3_nonthinking.jinja
  Qwen3-Embedding-0.6B/
  Qwen3-Reranker-0.6B/
```

## Start vLLM

Start vLLM in another terminal before running the pipeline:

```bash
export MODEL_NAME=Qwen3-8B
export VLLM_BATCH_INVARIANT=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve "$LLM_PATH/$MODEL_NAME" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size 4 \
  --enable-lora \
  --max-lora-rank 64 \
  --chat-template "$LLM_PATH/$MODEL_NAME/qwen3_nonthinking.jinja"
```

The code uses `http://localhost:8000/v1/chat/completions` and vLLM runtime LoRA loading.

## Run

From the repository root:

```bash
bash run.sh
```

The default example is `Qwen3-8B` on `Long-Long`. You can override the public knobs:

```bash
MODEL_NAME=Qwen3-8B \
DATA_TYPE=Long-Long \
OUTPUT_ROOT_PATH=./UNO/code/output_Qwen3-8B_Long-Long_uno_release \
SMALL_TEST=false \
bash run.sh
```

`run.sh` runs the full flow:

1. `main_init_test.py`: generate base/init test answers.
2. `main.py`: generate rules, revised answers, judged preference data, clusters, cognitive gap scores, Expert LoRAs, validation results, and primary-path test answers.
3. `main_reverse_user.py`: train Critic LoRAs and generate reflective-path answers.
4. `aggregate_uno_outputs.py`: merge final results into UNO-Single and UNO.

For a quick environment check:

```bash
SMALL_TEST=true bash run.sh
```

`SMALL_TEST=true` is not a paper-result reproduction.

## Final Outputs

For the default run:

```text
$HOME_PATH/UNO/code/output_Qwen3-8B_Long-Long_uno_release/Long-Long/cluster/
  uno_single_test_results.json
  uno_test_results.json
  uno_path_selection.json
```


- Choose a Primary epoch from `cluster_to_win_rate_by_bleu.json` when the best win rate is at least `0.53`.
- Mark a cluster as Reflective when `common_rules_info_scores.json` has `min_novelty_score >= 0.45`, or when Primary validation fails.
- UNO-Single uses Primary for selected Primary clusters and falls back to init for Reflective clusters.
- UNO uses Primary for selected Primary clusters and Reflective output for Reflective clusters.

## Paper Settings

- clustering distance threshold: `4`
- win-rate threshold: `0.53`
- BLEU threshold: `0.05`
- judge samples: `3`
- LoRA rank: `64`
- LoRA dropout: `0.05`
- training epochs: `8`
- learning rate: `5e-4`
- DPO/SFT loss weights: `0.5 / 0.5`

## Evaluation

Use the official MemoryBench evaluation workflow on `uno_single_test_results.json` and `uno_test_results.json`. This repository generates model responses. See [MemoryBench Evaluation Workflow](https://github.com/THUIR/MemoryBench) for details.

## Other Models

For Phi-4 or other models, first generate the corresponding user logs with MemoryBench, then keep `MODEL_NAME`, model paths, vLLM served name, and init-output paths consistent.
