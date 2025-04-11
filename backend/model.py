# %%

# from datasets import load_from_disk
import os
import glob

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from datasets import Dataset
import re
from tqdm import tqdm

from typing import List 


torch.set_float32_matmul_precision('high')


BATCH_SIZE = 256

# %%
# we need to create the mdm_list
# import the full mdm-only file
data_path = '../data/process/train.csv'
full_df = pd.read_csv(data_path)
entity_list = sorted(list((set(full_df['entity_name']))))


# %%
id2label = {}
label2id = {}
for idx, val in enumerate(entity_list):
    id2label[idx] = val
    label2id[val] = idx

# introduce pre-processing functions
def preprocess_text(text):
    # 1. Make all uppercase
    text = text.lower()

    # Substitute digits with 'x'
    # text = re.sub(r'\d+', '#', text)

    # standardize spacing
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# %%
# function to perform training for a given fold
def test(inputs: List[str]) -> List[str]:

    inputs = [preprocess_text(input) for input in inputs]
    test_dataset = Dataset.from_dict({'text': inputs})



    # prepare tokenizer

    checkpoint_directory = '../train/classification/checkpoint'
    # Use glob to find matching paths
    # path is usually checkpoint_fold_1/checkpoint-<step number>
    # we are guaranteed to save only 1 checkpoint from training
    pattern = 'checkpoint-*'
    model_checkpoint = glob.glob(os.path.join(checkpoint_directory, pattern))[0]

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt", clean_up_tokenization_spaces=True)


    # given a dataset entry, run it through the tokenizer
    def preprocess_function(example):
        input = example['text']
        # text_target sets the corresponding label to inputs
        # there is no need to create a separate 'labels'
        # and yes, it creates a new column "labels", not 'label'. wild.
        model_inputs = tokenizer(
            input,
            truncation=True,
            padding=False
        )
        return model_inputs

    # map maps function to each "row" in the dataset
    # aka the data in the immediate nesting
    datasets = test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns="text",
    )


    # datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(entity_list),
        id2label=id2label,
        label2id=label2id)
    # important! after extending tokens vocab
    model.resize_token_embeddings(len(tokenizer))

    model = model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    pred_ids = []

    # use the data collator to prepare collated batch (batch with equal token len)
    dataloader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
    for batch in tqdm(dataloader):
            # Inference in batches
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            

            # Move to GPU if available
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Perform inference
            with torch.no_grad():
                logits = model(
                    input_ids,
                    attention_mask).logits
                predicted_class_ids = logits.argmax(dim=1).to("cpu")
                pred_ids.extend(predicted_class_ids)

    pred_ids = [tensor.item() for tensor in pred_ids]
    y_pred = [id2label[id] for id in pred_ids]

    # clear memory before returning
    torch.cuda.empty_cache()
    return y_pred

