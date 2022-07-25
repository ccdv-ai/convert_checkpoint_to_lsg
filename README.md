
Under review

# Convert checkpoint to LSG

Transformers >= 4.18.0

This script converts any BERT/RoBERTa/CamemBERT/XLM-Roberta/DistilBert/BART/Pegasus checkpoint (from HuggingFace hub, [see](https://huggingface.co/ccdv)) to its LSG variant to handle long sequences. 

Use `convert_checkpoint.py`. Available model_type (inside config.json): 
* "bart" (encoder converted only)
* "barthez" (encoder converted only)
* "bert"
* "camembert"
* "distilbert"
* "mbart" (not tested extensively, encoder converted only)
* "pegasus" (not tested extensively, encoder converted only)
* "roberta"
* "xlm-roberta"

Model architecture is infered from config but you can specify a different one if the config is wrong (can happen for BART models), see  `python convert_bert_checkpoint.py --help`


BERT example (BertForPretraining):

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

RoBERTa example (RobertaForMaskedLM):
```bash
git clone https://github.com/ccdv-ai/convert_checkpoint_to_lsg.git
cd convert_checkpoint_to_lsg

export MODEL_TO_CONVERT=roberta-base
export MODEL_NAME=lsg-roberta-base
export MAX_LENGTH=4096

python convert_checkpoint.py \
    --initial_model $MODEL_TO_CONVERT \
    --model_name $MODEL_NAME \
    --model_kwargs "{'sparsity_type': 'lsh', 'block_size': 32}"
    --max_sequence_length $MAX_LENGTH
```

# Usage

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