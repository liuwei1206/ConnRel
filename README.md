# Annotation-Inspired Implicit Discourse Relation Classification with Auxiliary Discourse Connective Generation
Code for the ACL 2023 paper "Annotation-Inspired Implicit Discourse Relation Classification with Auxiliary Discourse Connective Generation"

## 1. Requirement
Our working environment is Python 3.8. Before you run the code, please make sure you have installed all the required packages. You can achieve it by simply execute the shell as `sh requirements.sh`

Then you need to download roberta-base from [here](https://huggingface.co/roberta-base/tree/main), and put it under the folder "data/pretrained_models/roberta-base".

## 2. Data and Preprocessing
For PDTB 2.0, copy the raw corpus under the folder "data/dataset/pdtb2/raw", and then do preprocessing via `python3 preprocessing`. (you may need to modify some easy codes in main function of preprocesing.py). The raw corpus looks like: 00, 01, 02, ..., 24.

For PDTB 3.0, copy the raw corpus under the folder "data/dataset/pdtb3/raw/gold" and "data/dataset/pdtb3/raw/data", where the former is label files and the latter is text files. Then, do preprocessing via `python3 preprocessing`. (you may need to modify some easy codes in main function of preprocesing.py). The corpus in both raw/gold and raw/data looks like: 00, 01, 02, ..., 24.

## 3. Run

