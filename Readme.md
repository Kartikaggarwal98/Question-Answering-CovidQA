# Out of domain Question Answering

This repo contains the code for out-of-domain question answering on covid question answering dataset using different models. Huggingface library is used to run all models on ```deepset/roberta-base-squad2``` model. The following models are used:

1. Zero-shot inference: In order to build baseline for further models, zero-shot inference using HF question-answring pipeline is done. This gave an Exact Match of 24.2 on Test set. Run ```roberta_zeroshot.py``` for inference.

1. Finetuning: The model is finetuned on covid dataset for 3 epochs. This gave Exact Match of 35.4 on Test set. Run ```roberta_finetune.py``` for training and inference.

1. Adapter Model: Instead of finetuning complete model, adapter layers are used which reduce the learnable parameters by a large amount (130X). See AdapterHub for more information. When trained on 3 epochs, the exact match on test is 33.8. Run ```roberta_adapter.py``` for training and inference.