API_KEY=$API_KEY
model=gpt-4-0613
output_dir=$MODEL_PATH
reference_model=$MODEL1
target_model=$MODEL2

python evaluation_gpt4.py \
    --API_KEY $API_KEY \
    --model $model \
    --qa_file $DATA_PATH \
    --key_1 $reference_model \
    --key_2 $target_model \
    --output_dir $output_dir \

python evaluation_gpt4.py \
    --API_KEY $API_KEY \
    --model $model \
    --qa_file $DATA_PATH \
    --key_1 $target_model \
    --key_2 $reference_model \
    --output_dir $output_dir \
