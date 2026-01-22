HOME_PATH=$7
LLM_PATH=$8
OUTPUT_DIR=$9
BETA=${10}
LR=${11}
model_name=${12}

NUM=$6
project_path=$HOME_PATH/UNO/code/train
model_path=$LLM_PATH/${model_name}
port_addr=11470
train_file_path=$1
train_key=$2
cluster_path=$3
sigmoid_weight=$4
sft_weight=$5

unset NCCL_NET_PLUGIN


export DS_SKIP_CUDA_CHECK=1
export CXX=g++
output_path=${OUTPUT_DIR}/${train_key}_${model_name}_sigmoid${sigmoid_weight}_sft${sft_weight}

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port=$port_addr --num_gpus=${NUM} $project_path/dpo_sft_train.py \
    --report_to "tensorboard" \
    --data_path $train_file_path \
    --cluster_key $train_key \
    --cluster_path $cluster_path \
    --model_name_or_path $model_path \
    --output_dir $output_path \
    --max_length 12000 \
    --loss_type sigmoid sft \
    --loss_weights $sigmoid_weight $sft_weight \
    --beta $BETA \
    --num_train_epochs 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy epoch \
    --learning_rate $LR \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
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