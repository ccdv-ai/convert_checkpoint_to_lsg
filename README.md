# LSG Attention: Extrapolation of pretrained Transformers to long sequences

Under review

Requires `transformers >= 4.22.0`

* [Compatible models](#compatible-models)
* [Efficiency](#efficiency)
* [Conversion](#convert-checkpoint-to-lsg)
* [Usage](#model-usage)
* [LSG-Attention](#lsg-attention)
* [Experiments](#experiments)

# Compatible models

This script converts HuggingFace checkpoints to its LSG (Local-Sparse-Global) variant to handle long sequences. Available model types:

* ALBERT [`"albert"`]
* BART [`"bart"`] (encoder attention modified only)
* BARTHEZ [`"barthez"`] (encoder attention modified only)
* BERT [`"bert"`]
* CamemBERT [`"camembert"`]
* DistilBERT [`"distilbert"`]
* Electra [`"electra"`]
* mBART [`"mbart"`] (not tested extensively, encoder attention modified only)
* Pegasus [`"pegasus"`] (not tested extensively, encoder attention modified only)
* RoBERTa [`"roberta"`]
* XLM-RoBERTa [`"xlm-roberta"`]

Some converted checkpoints are available [here](https://huggingface.co/ccdv).

![attn](img/attn.png)

# Efficiency
Memory and training speed for a binary classification task with 4x4096 tokens batches and Adam (RTX 8000).

| Models                     | Seconds per step | Memory w/ attn dropout  | Memory w/o attn dropout |
|----------------------------|------------------|-------------------------|-------------------------|
| Longformer-base            | 3.22 s/step      | 34.38 Gb                | 32.83 Gb                |
| BigBird-RoBERTa-base       | 2.85 s/step      | 38.13 Gb                | 38.13 Gb (no effect)    |
| LSG-RoBERTa 256/0          | 1.40 s/step      | 32.92 Gb                | 24.80 Gb                |
| LSG-RoBERTa 128/128 (norm) | 1.51 s/step      | 33.80 Gb                | 27.52 Gb                |
| LSG-RoBERTa 32/32 (norm)   | 1.20 s/step      | 24.53 Gb                | 22.53 Gb                |



# Convert checkpoint to LSG 

Use `convert_checkpoint.py` to convert a model (check model_type from config.json). \
The architecture of the model is inferred from the config file, but you can specify a different one if the config is incorrect (which can happen for BART models), see  `python convert_checkpoint.py --help`. \
To test the converted model, add `--run_test` (experimental).


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

LSG relies on 3 components: block local attention, sparse attention and prepended global attention.

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
If `sparse_block_size=0` or `sparsity_type="none"`, only local attention is considered. \
Note that for sequences with length < 2*block_size, the type has no effect.
* `sparsity_type="norm"`, select highest norm tokens
    * Works best for a small sparsity_factor (2 to 4)
    * Additional parameters:
        * None
* `sparsity_type="pooling"`, use average pooling to merge tokens
    * Works best for a small sparsity_factor (2 to 4)
    * Additional parameters:
        * None
* `sparsity_type="lsh"`, use the LSH algorithm to cluster similar tokens
    * Works best for a large sparsity_factor (4+)
    * LSH relies on random projections, thus inference may differ slightly with different seeds
    * Additional parameters:
        * lsg_num_pre_rounds=1, pre merge tokens n times before computing centroids
* `sparsity_type="stride"`, use a striding mecanism per head
    * Each head will use different tokens strided by sparsify_factor
    * Not recommended if sparsify_factor > num_heads
* `sparsity_type="block_stride"`, use a striding mecanism per head
    * Each head will use block of tokens strided by sparsify_factor
    * Not recommended if sparsify_factor > num_heads

# Experiments

See `experiments/`