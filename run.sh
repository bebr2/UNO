

model_name=Qwen3-8B
epsilon="0.53"
output_root_path=./UNO/code/your_output_dir
data_type="Long-Long"
sft_weight=0.5
sigmoid_weight=0.5

echo "output_root_path: $output_root_path"


python main.py --model $model_name --output_root_path $output_root_path --data_type ${data_type} --sft_weight $sft_weight --sigmoid_weight $sigmoid_weight --judge_generate_num 3 --epsilon ${epsilon}



python main_reverse.py --root_path $output_root_path \
--model $model_name \
--data_type $data_type \
--train_output_dir reverse_train
