HOME_PATH=$5
LLM_PATH=$6
train_key=$2
cluster_path=$3
model_name=$8
data_path=$1
OUTPUT_DIR=$7
project_path=$HOME_PATH/UNO/code/train
model_path=$LLM_PATH/${model_name}
port_addr=11470

NUM=$4

output_path=${OUTPUT_DIR}/${train_key}_${model_name}
unset NCCL_NET_PLUGIN


export DS_SKIP_CUDA_CHECK=1
export CXX=g++


CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port=$port_addr --num_gpus=${NUM} $project_path/sft_rules.py \
    --report_to "tensorboard" \
    --data_path $data_path \
    --cluster_key $train_key \
    --cluster_path $cluster_path \
    --model_name_or_path $model_path \
    --output_dir $output_path \
    --max_length 8000 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy epoch \
    --learning_rate 5e-4 \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --eval_strategy epoch \
    --do_eval True \
    --metric_for_best_model "eval_loss" \
    --load_best_model_at_end True \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed $project_path/ds_config.json \
    --seed 42 \
    --bf16 True 

for ckpt in ${output_path}/checkpoint-*; do
  if [ -d "$ckpt" ]; then
    rm -rf $ckpt/global_step*
    echo "Cleaned state in $ckpt"
  fi
done