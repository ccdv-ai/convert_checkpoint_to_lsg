export MODEL_NAME=ccdv/lsg-base-4096
export DATASET_NAME=imdb
export MAX_SEQ_LENGTH=4096
export MODEL_PREFIX=tmp/imdb/lsg-base

python run_classification.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name $DATASET_NAME  \
  --do_train \
  --do_eval \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --save_strategy no \
  --eval_with_test True \
  --model_kwargs "{'num_global_tokens': 1, 'sparsity_type': 'norm', 'sparsity_factor': 4, 'sparse_block_size': 32, 'block_size': 32}" \
  --shuffle True \
  --output_dir $MODEL_PREFIX