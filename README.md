
Under review

Requires `transformers >= 4.18.0`

This script converts any AlBERT/BERT/RoBERTa/CamemBERT/XLM-Roberta/DistilBert/BART/Pegasus checkpoint (from HuggingFace hub, [see](https://huggingface.co/ccdv)) to its LSG variant to handle long sequences.

* [Conversion](#convert-checkpoint-to-lsg)
* [Usage](#model-usage)
* [LSG-Attention](#lsg-attention)

![attn](img/attn.png)

# Convert checkpoint to LSG 

Use `convert_checkpoint.py` with these model types (model_type from config.json): 
* "albert"
* "bart" (encoder attention modified only)
* "barthez" (encoder attention modified only)
* "bert"
* "camembert"
* "distilbert"
* "mbart" (not tested extensively, encoder attention modified only)
* "pegasus" (not tested extensively, encoder attention modified only)
* "roberta"
* "xlm-roberta"

Model architecture is infered from config but you can specify a different one if the config is wrong (can happen for BART models), see  `python convert_checkpoint.py --help`


BERT example (`BertForPretraining`):

```bash
git clone https://github.com/ccdv-ai/convert_checkpoint_to_lsg.git
cd convert_checkpoint_to_lsg

export MODEL_TO_CONVERT=bert-base-uncased
export MODEL_NAME=lsg-bert-base-uncased
export MAX_LENGTH=4096

python convert_checkpoint.py \
    --initial_model $MODEL_TO_CONVERT \
    --model_name $MODEL_NAME \
    --max_sequence_length $MAX_LENGTH
```

RoBERTa example (from `RobertaForMaskedLM` to `RobertaForSequenceClassification`):
```bash
git clone https://github.com/ccdv-ai/convert_checkpoint_to_lsg.git
cd convert_checkpoint_to_lsg

export MODEL_TO_CONVERT=roberta-base
export MODEL_NAME=lsg-roberta-base
export MAX_LENGTH=4096

python convert_checkpoint.py \
    --initial_model $MODEL_TO_CONVERT \
    --model_name $MODEL_NAME \
    --model_kwargs "{'mask_first_token': true, 'sparsity_type': 'lsh', 'block_size': 32}" \
    --architecture RobertaForSequenceClassification \
    --max_sequence_length $MAX_LENGTH
```

# Model Usage

Works with the AutoClass.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load created model
MODEL_NAME = "lsg-roberta-base"
SENTENCE = "This is a test sentence."

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

inputs = tokenizer(SENTENCE, return_tensors="pt")
model(**inputs)
```

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load created model
MODEL_NAME = "lsg-roberta-base"
SENTENCE = "This is a test sentence."

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

inputs = tokenizer(SENTENCE, return_tensors="pt")
model(**inputs)
```

# LSG Attention

## Parameters
You can change various parameters like : 
* local block size (block_size=128)
* sparse block size (sparse_block_size=128)
* sparsity factor (sparsity_factor=2)
* mask_first_token (mask first token since it is redundant with the first global token)
* the number of global tokens (num_global_tokens=1)
* see config.json file

## Sparse selection type
There are 5 different sparse selection patterns. The best type is task dependent. \
Note that for sequences with length < 2*block_size, the type has no effect.
* sparsity_type="norm", select highest norm tokens
    * Works best for a small sparsity_factor (2 to 4)
    * Additional parameters:
        * None
* sparsity_type="pooling", use average pooling to merge tokens
    * Works best for a small sparsity_factor (2 to 4)
    * Additional parameters:
        * None
* sparsity_type="lsh", use the LSH algorithm to cluster similar tokens
    * Works best for a large sparsity_factor (4+)
    * LSH relies on random projections, thus inference may differ slightly with different seeds
    * Additional parameters:
        * lsg_num_pre_rounds=1, pre merge tokens n times before computing centroids
* sparsity_type="stride", use a striding mecanism per head
    * Each head will use different tokens strided by sparsify_factor
    * Not recommended if sparsify_factor > num_heads
* sparsity_type="block_stride", use a striding mecanism per head
    * Each head will use block of tokens strided by sparsify_factor
    * Not recommended if sparsify_factor > num_heads