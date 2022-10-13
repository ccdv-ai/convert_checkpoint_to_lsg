export MODEL_NAME=ccdv/lsg-legal-base-uncased-4096
export MAX_SEQ_LENGTH=512
export MODEL_PREFIX=tmp/lex_glue/ecthr_a/lsg-bert-legal
export TASK_NAME=ecthr_a

python run_lexglue.py \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --save_strategy no \
  --eval_with_test True \
  --shuffle \
  --output_dir $MODEL_PREFIX

