#!/usr/bin/env bash
set -e

# Before running this script, start vLLM in another terminal. Example:
#
# export MODEL_NAME=Qwen3-8B
# export VLLM_BATCH_INVARIANT=1
# export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve "$LLM_PATH/$MODEL_NAME" \
#   --served-model-name "$MODEL_NAME" \
#   --tensor-parallel-size 4 \
#   --enable-lora \
#   --max-lora-rank 64 \
#   --chat-template "$LLM_PATH/$MODEL_NAME/qwen3_nonthinking.jinja"

MODEL_NAME=${MODEL_NAME:-Qwen3-8B}
DATA_TYPE=${DATA_TYPE:-Long-Long}
OUTPUT_ROOT_PATH=${OUTPUT_ROOT_PATH:-./UNO/code/output_${MODEL_NAME}_${DATA_TYPE}_uno_release}
INIT_OUTPUT_ROOT_PATH=${INIT_OUTPUT_ROOT_PATH:-./UNO/code/output_init_0}

EPSILON=${EPSILON:-0.53}
BLEU_RATIO=${BLEU_RATIO:-0.05}
SFT_WEIGHT=${SFT_WEIGHT:-0.5}
SIGMOID_WEIGHT=${SIGMOID_WEIGHT:-0.5}
JUDGE_GENERATE_NUM=${JUDGE_GENERATE_NUM:-3}
DISTANCE_THRESHOLD=${DISTANCE_THRESHOLD:-4}
SMALL_TEST=${SMALL_TEST:-false}

: "${HOME_PATH:?Set HOME_PATH to the parent directory of UNO.}"
: "${LLM_PATH:?Set LLM_PATH to the model root directory.}"
: "${MEMORYBENCH_PATH:?Set MEMORYBENCH_PATH to the local MemoryBench dataset path.}"
: "${CONFIG_PATH:?Set CONFIG_PATH to MemoryBench configs/datasets.}"

SMALL_TEST_ARG=""
AGGREGATE_SMALL_TEST_ARG=""
if [ "$SMALL_TEST" = "true" ]; then
  SMALL_TEST_ARG="--small_test"
  AGGREGATE_SMALL_TEST_ARG="--allow_epoch0_fallback"
fi

cd "$(dirname "$0")/code"

echo "Step 1/4: generate base/init test answers"
python main_init_test.py \
  --model "$MODEL_NAME" \
  --output_root_path "$INIT_OUTPUT_ROOT_PATH" \
  --data_type "$DATA_TYPE" \
  --config_path "$CONFIG_PATH" \
  $SMALL_TEST_ARG

echo "Step 2/4: build preprocessing artifacts, clusters, cognitive gap scores, Expert LoRAs, and primary-path answers"
python main.py \
  --model "$MODEL_NAME" \
  --output_root_path "$OUTPUT_ROOT_PATH" \
  --data_type "$DATA_TYPE" \
  --config_path "$CONFIG_PATH" \
  --distance_threshold "$DISTANCE_THRESHOLD" \
  --sft_weight "$SFT_WEIGHT" \
  --sigmoid_weight "$SIGMOID_WEIGHT" \
  --judge_generate_num "$JUDGE_GENERATE_NUM" \
  --epsilon "$EPSILON" \
  --bleu_ratio "$BLEU_RATIO" \
  $SMALL_TEST_ARG

echo "Step 3/4: build Critic LoRAs and reflective-path answers"
python main_reverse_user.py \
  --model "$MODEL_NAME" \
  --root_path "$OUTPUT_ROOT_PATH" \
  --data_type "$DATA_TYPE" \
  --config_path "$CONFIG_PATH" \
  --train_output_dir reverse_train \
  $SMALL_TEST_ARG

echo "Step 4/4: aggregate final UNO-Single and UNO outputs"
python aggregate_uno_outputs.py \
  --model "$MODEL_NAME" \
  --data_type "$DATA_TYPE" \
  --output_root_path "$OUTPUT_ROOT_PATH" \
  --init_root_path "$INIT_OUTPUT_ROOT_PATH" \
  --win_rate_threshold "$EPSILON" \
  --novel_min_score 0.45 \
  $AGGREGATE_SMALL_TEST_ARG

echo "Done. Final files are in:"
echo "$HOME_PATH/$OUTPUT_ROOT_PATH/$DATA_TYPE/cluster/uno_single_test_results.json"
echo "$HOME_PATH/$OUTPUT_ROOT_PATH/$DATA_TYPE/cluster/uno_test_results.json"
