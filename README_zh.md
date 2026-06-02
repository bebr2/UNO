# UNO: User log-driveN Optimization

UNO 是一个使用用户日志优化 LLM 系统的框架。它会从用户反馈中提取半结构化规则和偏好数据，按 query-rule 语义聚类，估计模型和用户日志之间的 cognitive gap，并在测试时为不同簇选择 Primary 或 Reflective 经验模块。

英文版文档见 [README.md](README.md)。

## 代码对应关系

| 论文模块 | 代码入口 | 作用 |
| --- | --- | --- |
| Preprocessing | `generate_rules.py`, `revise_answer_multiple.py`, `judge.py`, `main.py` | 提取反馈规则、生成修订答案、judge 打分、构造偏好数据。 |
| Query-and-feedback clustering | `cluster_a.py`, `main.py` | 使用 query 和 rule embedding 做聚类。 |
| Cognitive gap assessment | `hinfo.py`, `main.py` | 使用 Qwen3-Reranker-0.6B 判断簇是否属于 high-gap。 |
| Primary Experience Module | `main.py`, `train/dpo_sft.sh` | 训练每个簇的 Expert LoRA，损失为 DPO + SFT。 |
| Reflective Experience Module | `main_reverse_user.py`, `get_rule_train_data.py`, `train/sft.sh` | 训练 Critic LoRA，用来生成修改建议。 |
| 输出整合 | `aggregate_uno_outputs.py` | 将 init、primary、reflective 输出整合成 UNO-Single 和 UNO。 |

## 环境

```bash
conda create -n uno python=3.11.14
conda activate uno
pip install -r requirements.txt
```

需要本地准备：

- `Qwen3-8B`
- `Qwen3-Embedding-0.6B`
- `Qwen3-Reranker-0.6B`
- `qwen3_nonthinking.jinja`
- MemoryBench 数据集
- MemoryBench 的 `configs/datasets`

## 路径

运行前设置：

```bash
export HOME_PATH=/path/to/parent/of/UNO
export LLM_PATH=/path/to/model_root
export MEMORYBENCH_PATH=/path/to/MemoryBench
export CONFIG_PATH=/path/to/MemoryBench/configs/datasets
```

注意：`HOME_PATH` 是 UNO 仓库的父目录，不是 UNO 仓库本身。例如仓库在 `/data/code/UNO`，则设置 `HOME_PATH=/data/code`。

模型目录建议如下：

```text
$LLM_PATH/
  Qwen3-8B/
    qwen3_nonthinking.jinja
  Qwen3-Embedding-0.6B/
  Qwen3-Reranker-0.6B/
```

## 启动 vLLM

先在另一个终端启动 vLLM：

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

代码默认访问 `http://localhost:8000/v1/chat/completions`，并使用 vLLM 的 runtime LoRA 加载接口。

## 运行

在仓库根目录执行：

```bash
bash run.sh
```

默认示例是 `Qwen3-8B` + `Long-Long`。可以覆盖这些变量：

```bash
MODEL_NAME=Qwen3-8B \
DATA_TYPE=Long-Long \
OUTPUT_ROOT_PATH=./UNO/code/output_Qwen3-8B_Long-Long_uno_release \
SMALL_TEST=false \
bash run.sh
```

`run.sh` 从头到尾执行四步：

1. `main_init_test.py`：生成 base/init 测试答案。
2. `main.py`：生成规则、修订答案、judge 后的偏好数据、聚类、cognitive gap 分数、Expert LoRA、验证结果和 primary-path 测试答案。
3. `main_reverse_user.py`：训练 Critic LoRA，并生成 reflective-path 测试答案。
4. `aggregate_uno_outputs.py`：整合最终 UNO-Single 和 UNO 输出。

快速检查环境可以运行：

```bash
SMALL_TEST=true bash run.sh
```

`SMALL_TEST=true` 只用于检查流程，不是论文结果复现。

## 最终输出

默认运行后，最终文件在：

```text
$HOME_PATH/UNO/code/output_Qwen3-8B_Long-Long_uno_release/Long-Long/cluster/
  uno_single_test_results.json
  uno_test_results.json
  uno_path_selection.json
```

- 若 `cluster_to_win_rate_by_bleu.json` 中某簇最佳 win rate 不低于 `0.53`，该簇通过 Primary verifier，并选择对应 epoch。
- 若 `common_rules_info_scores.json` 中某簇 `min_novelty_score >= 0.45`，该簇视为 high-gap / reflective。
- Primary 验证失败的簇也进入 Reflective 路径。
- UNO-Single：Primary 簇使用 Expert LoRA 输出，Reflective 簇回退 init。
- UNO：Primary 簇使用 Expert LoRA 输出，Reflective 簇使用 Critic LoRA + revise 的 reflective 输出。

## 论文设置


- 聚类距离阈值：`4`
- win-rate 阈值：`0.53`
- BLEU 阈值：`0.05`
- judge 采样次数：`3`
- LoRA rank：`64`
- LoRA dropout：`0.05`
- 训练 epoch：`8`
- learning rate：`5e-4`
- DPO/SFT loss weights：`0.5 / 0.5`

## 评估

对 `uno_single_test_results.json` 和 `uno_test_results.json` 使用 MemoryBench 官方评估流程。见 [MemoryBench 评估流程](https://github.com/THUIR/MemoryBench)。

## 其他模型

如果使用 Phi-4 或其他模型，需要先按 MemoryBench 生成对应模型的用户日志，并保证 `MODEL_NAME`、模型路径、vLLM served name、init 输出路径一致。
