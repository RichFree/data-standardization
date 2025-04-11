# %%

# from datasets import load_from_disk
import os

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import (
    T5TokenizerFast,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments
)
import evaluate
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
import re

torch.set_float32_matmul_precision('high')

# %%
def _preprocess_text(text):
    # 1. Make all uppercase
    text = text.lower()
    # Substitute digits with 'x'
    # text = re.sub(r'\d+', '#', text)
    # standardize spacing
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# outputs a list of dictionaries
def process_df_to_dict(df):
    output_list = []
    for _, row in df.iterrows():
        desc = row['mention']
        entity_name = row['entity_name']
        element = {
            'input' : _preprocess_text(desc),
            'output': entity_name,

        }
        output_list.append(element)

    return output_list


# function to perform training for a given data
def train():

    save_path = f'checkpoint'

    data_path = f"../../data/process/train.csv"
    train_df = pd.read_csv(data_path)
    train_dataset = Dataset.from_list(process_df_to_dict(train_df))

    # prepare tokenizer
    model_checkpoint = "t5-base"
    tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint, return_tensors="pt", clean_up_tokenization_spaces=True)


    # given a dataset entry, run it through the tokenizer
    def preprocess_function(example):
        input = example['input']
        target = example['output']
        # text_target sets the corresponding label to inputs
        # there is no need to create a separate 'labels'
        model_inputs = tokenizer(
            input,
            text_target=target, 
            truncation=False,
            padding=False
        )
        return model_inputs

    # map maps function to each "row" in the dataset
    # aka the data in the immediate nesting
    tokenized_datasets = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        # we only want the tokens, not the string
        remove_columns=train_dataset.column_names,
    )

    # https://github.com/huggingface/transformers/pull/28414
    # model_checkpoint = "google/t5-efficient-tiny"
    # device_map set to auto to force it to load contiguous weights 
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, device_map='auto')

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    # important! after extending tokens vocab
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = evaluate.load("sacrebleu")


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, 
                                            skip_special_tokens=False)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=False)

        # Remove <PAD> tokens from decoded predictions and labels
        decoded_preds = [pred.replace(tokenizer.pad_token, '').strip() for pred in decoded_preds]
        decoded_labels = [[label.replace(tokenizer.pad_token, '').strip()] for label in decoded_labels]

        # Some simple post-processing
        # decoded_preds = [pred.strip() for pred in decoded_preds]
        # decoded_labels = [[label.strip()] for label in decoded_labels]
        # print(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}


    # Generation Config
    # from transformers import GenerationConfig
    gen_config = model.generation_config
    gen_config.max_length = 90 # slightly reduce the generation length

    # compile
    # model = torch.compile(model, backend="inductor", dynamic=True)


    # Trainer

    args = Seq2SeqTrainingArguments(
        f"{save_path}",
        # eval_strategy="epoch",
        eval_strategy="no",
        logging_dir="tensorboard-log",
        logging_strategy="epoch",
        # save_strategy="epoch",
        load_best_model_at_end=False,
        learning_rate=1e-3,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        auto_find_batch_size=False,
        ddp_find_unused_parameters=False,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=80,
        predict_with_generate=True,
        bf16=True,
        push_to_hub=False,
        generation_config=gen_config,
        remove_unused_columns=False,
    )


    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets,
        # eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # uncomment to load training from checkpoint
    # checkpoint_path = 'default_40_1/checkpoint-5600'
    # trainer.train(resume_from_checkpoint=checkpoint_path)

    trainer.train()

# execute training
train()

