All prompt templates in this project are in the `prompts.py` file.

## Getting Started

### Environment Setup

create a virtual environment:

```bash
conda create -n uno python=3.11.14
pip install -r requirements.txt
```

Download the `configs` directory from [MemoryBench](https://github.com/LittleDinoC/MemoryBench/) and the [MemoryBench-Dataset](https://huggingface.co/datasets/THUIR/MemoryBench) from Hugging Face to your local machine. Then, update all `path/to/xxx` placeholders in this project to match your local paths.

### Qwen3-8B Example

Before proceeding, set the `LLM_PATH` and `HOME_PATH` environment variables. `LLM_PATH` should point to the directory containing the large language model, while `HOME_PATH` should point to the parent directory of this project. For example:

```bash
export LLM_PATH=path/to/your/llm_directory
export HOME_PATH=path/to/your/UNO_directory
```

Make sure to place [qwen3_nonthinking.jinja](https://github.com/QwenLM/Qwen3/blob/main/docs/source/assets/qwen3_nonthinking.jinja) in the `${LLM_PATH}` directory.

Start the vLLM service:

```bash
cd ${LLM_PATH}
export VLLM_BATCH_INVARIANT=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen3-8B --tensor_parallel_size 4 --enable-lora --max-lora-rank 64 --chat-template qwen3_nonthinking.jinja
```

Then execute:

```bash
source run.sh
```

Note that this project aims to reproduce reported performance and therefore constructs both the `Primary Experience Module` and the `Reflective Experience Module` for all clusters. In practical scenarios, clusters that successfully pass the `Primary Experience Evaluation` and `Cognitive Gap Assessment` do not require the construction of the `Reflective Experience Module`.

### Phi-4 Example

If you use Phi-4 or other models, first generate the corresponding user logs following the instructions provided in [MemoryBench](https://github.com/LittleDinoC/MemoryBench/).

### Evaluation

Follow the evaluation procedure defined in [MemoryBench](https://github.com/LittleDinoC/MemoryBench/).