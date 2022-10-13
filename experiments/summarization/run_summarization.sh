export MODEL_NAME=ccdv/lsg-bart-base-4096
export DATASET_NAME=scientific_papers
export DATASET_CONFIG_NAME=arxiv
export MAX_SOURCE_LENGTH=4096
export MAX_TARGET_LENGTH=512
export MODEL_PREFIX=tmp/arxiv/lsg-bart

python run_summarization.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name $DATASET_NAME  \
  --dataset_config_name $DATASET_CONFIG_NAME \
  --do_train \
  --max_source_length $MAX_SOURCE_LENGTH \
  --max_target_length $MAX_TARGET_LENGTH \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 8e-5 \
  --num_train_epochs 6 \
  --save_strategy no \
  --model_kwargs "{'sparsity_type': 'none', 'sparsity_factor': 4, 'sparse_block_size': 0, 'block_size': 256}" \
  --shuffle \
  --warmup_ratio 0.1 \
  --output_dir $MODEL_PREFIX