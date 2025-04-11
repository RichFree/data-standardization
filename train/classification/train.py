# %%

# from datasets import load_from_disk
import os

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    EarlyStoppingCallback,
    TrainingArguments
)
import evaluate
import numpy as np
import pandas as pd
import re
# import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict

torch.set_float32_matmul_precision('high')

# %%
print(torch.cuda.is_available())

# %%
# we need to create the mdm_list
# import the full mdm-only file
data_path = '../../data/process/train.csv'
full_df = pd.read_csv(data_path)
entity_list = sorted(list((set(full_df['entity_name']))))


# %%
id2label = {}
label2id = {}
for idx, val in enumerate(entity_list):
    id2label[idx] = val
    label2id[val] = idx

# %%

# introduce pre-processing functions
def preprocess_text(text):
    # 1. Make all uppercase
    text = text.lower()

    # Substitute digits with 'x'
    # text = re.sub(r'\d+', '#', text)

    # standardize spacing
    text = re.sub(r'\s+', ' ', text).strip()

    return text



# outputs a list of dictionaries
# processes dataframe into lists of dictionaries
# each element maps input to output
# input: tag_description
# output: class label
def process_df_to_dict(df, mdm_list):
    output_list = []
    for _, row in df.iterrows():
        desc = f"{row['mention']}"
        pattern = f"{row['entity_name']}"
        try:
            index = mdm_list.index(pattern)
        except ValueError:
            print("Error: value not found in MDM list")
            index = -1
        element = {
            'text' : preprocess_text(desc),
            'label': index,
        }
        output_list.append(element)

    return output_list



# %%
# function to perform training for a given fold
def train():

    save_path = f'checkpoint'

    # training data 
    data_path = f"../../data/process/train.csv"
    train_df = pd.read_csv(data_path)

    # for now no validation
    train_dataset = Dataset.from_list(process_df_to_dict(train_df, entity_list))
        # 'validation' : Dataset.from_list(process_df_to_dict(validation_df, entity_list)),


    # prepare tokenizer
    # model_checkpoint = "distilbert/distilbert-base-uncased"
    model_checkpoint = 'google-bert/bert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt", clean_up_tokenization_spaces=True)


    # given a dataset entry, run it through the tokenizer
    def preprocess_function(example):
        input = example['text']
        # text_target sets the corresponding label to inputs
        # there is no need to create a separate 'labels'
        model_inputs = tokenizer(
            input,
            truncation=True,
            padding=False
        )
        return model_inputs

    # map maps function to each "row" in the dataset
    # aka the data in the immediate nesting
    tokenized_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns="text",
    )

    # create data collator
    # pad to fixed length in order for compile to work
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=32, padding="max_length", return_tensors='pt')
    # pad to per-batch max_length for dynamic execution without compile
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # compute metrics
    metric = evaluate.load("accuracy")


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)

    # create id2label and label2id
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(entity_list),
        id2label=id2label,
        label2id=label2id)
    # important! after extending tokens vocab
    model.resize_token_embeddings(len(tokenizer))


    #
    # Trainer
    training_args = TrainingArguments(
        output_dir=f"{save_path}",
        # eval_strategy="epoch",
        eval_strategy="no",
        logging_dir="tensorboard-log",
        logging_strategy="epoch",
        # save_strategy="epoch",
        load_best_model_at_end=False,
        learning_rate=1e-4,
        per_device_train_batch_size=128,
        # per_device_eval_batch_size=256,
        auto_find_batch_size=False,
        ddp_find_unused_parameters=False,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=80,
        bf16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        # torch_compile=True,
    )


    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset,
        # eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # uncomment to load training from checkpoint
    # checkpoint_path = 'default_40_1/checkpoint-5600'
    # trainer.train(resume_from_checkpoint=checkpoint_path)

    trainer.train()

# %%
# execute training
train()
# %%
