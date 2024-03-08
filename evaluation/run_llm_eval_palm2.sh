gcloud auth application-default set-quota-project instruction-tuning

output_dir=$MODEL_PATH
reference_model=$MODEL1
target_model=$MODEL2

python evaluation_palm2.py \
    --qa_file $DATA_PATH \
    --key_1 $reference_model \
    --key_2 $target_model \
    --output_dir $output_dir \

python evaluation_palm2.py \
    --qa_file $DATA_PATH \
    --key_1 $target_model \
    --key_2 $reference_model \
    --output_dir $output_dir \
