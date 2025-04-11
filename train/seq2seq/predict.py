import torch
from torch.utils.data import DataLoader
from transformers import (
    T5TokenizerFast,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
import os
from tqdm import tqdm
from datasets import Dataset
import numpy as np
import pandas as pd
import glob
import re

# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

BATCH_SIZE = 128
MAX_GENERATE_LENGTH = 90

# %%
class Inference():
    tokenizer: T5TokenizerFast
    model: torch.nn.Module
    dataloader: DataLoader

    def __init__(self, checkpoint_path):
        self._create_tokenizer()
        self._load_model(checkpoint_path)


    def _create_tokenizer(self):
        # load tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-base", return_tensors="pt", clean_up_tokenization_spaces=True)
        # Define additional special tokens

    def _load_model(self, checkpoint_path: str):
        # load model
        # Define the directory and the pattern
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        model = torch.compile(model)
        # set model to eval
        self.model = model.eval()


    def prepare_dataloader(self, input_df, batch_size):
        """
        *arguments*
        - input_df: input dataframe containing fields 'tag_description', 'thing', 'property'
        - batch_size: the batch size of dataloader output
        - max_length: length of tokenizer output
        """
        print("preparing dataloader")
        # convert each dataframe row into a dictionary
        # outputs a list of dictionaries

        def _preprocess_text(text):
            # 1. Make all uppercase
            text = text.lower()

            # Substitute digits with 'x'
            # text = re.sub(r'\d+', '#', text)

            # standardize spacing
            text = re.sub(r'\s+', ' ', text).strip()

            return text


        def _process_df(df):
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

        def _preprocess_function(example):
            input = example['input']
            target = example['output']
            # text_target sets the corresponding label to inputs
            # there is no need to create a separate 'labels'
            model_inputs = self.tokenizer(
                input,
                text_target=target, 
                truncation=False,
                padding=False,
            )
            return model_inputs

        test_dataset = Dataset.from_list(_process_df(input_df))


        # map maps function to each "row" in the dataset
        # aka the data in the immediate nesting
        datasets = test_dataset.map(
            _preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=test_dataset.column_names,
        )

        # use a data_collator for proper batch padding
        # create dataloader
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        self.dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=False, collate_fn=data_collator)


    def generate(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pred_generations = []

        print("start generation")
        for batch in tqdm(self.dataloader):
            # Inference in batches
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Move to GPU if available
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            self.model.to(device)

            # Perform inference
            with torch.no_grad():
                outputs = self.model.generate(input_ids,
                                        attention_mask=attention_mask,
                                        max_length=MAX_GENERATE_LENGTH)
                
                # Decode the output and print the results
                pred_generations.extend(outputs.to("cpu"))



        # # extract sequence and decode
        # def extract_seq(tokens, start_value, end_value):
        #     if start_value not in tokens or end_value not in tokens:
        #         return None  # Or handle this case according to your requirements
        #     start_id = np.where(tokens == start_value)[0][0]
        #     end_id = np.where(tokens == end_value)[0][0]
        #     return tokens[start_id+1:end_id]

        def process_tensor_output(tokens):
            # since there is only 1 field
            # there is no need to extract, just decode directly
            return self.tokenizer.decode(tokens, skip_special_tokens=True)



        # def process_tensor_output(tokens):
        #     thing_seq = extract_seq(tokens, 32100, 32101) # 32100 = <THING_START>, 32101 = <THING_END>
        #     property_seq = extract_seq(tokens, 32102, 32103) # 32102 = <PROPERTY_START>, 32103 = <PROPERTY_END>
        #     p_thing = None
        #     p_property = None
        #     if (thing_seq is not None):
        #         p_thing =  self.tokenizer.decode(thing_seq, skip_special_tokens=False)
        #     if (property_seq is not None):
        #         p_property =  self.tokenizer.decode(property_seq, skip_special_tokens=False)
        #     return p_thing, p_property

        # decode prediction labels
        def decode_preds(tokens_list):
            entity_prediction_list = []
            for tokens in tokens_list:
                entity_prediction = process_tensor_output(tokens)
                entity_prediction_list.append(entity_prediction)
            return entity_prediction_list 

        entity_prediction_list = decode_preds(pred_generations)
        return entity_prediction_list


# driver code
def main():
    data_path = f"../../data/process/test.csv"
    test_df = pd.read_csv(data_path)

    # import checkpoint
    checkpoint_directory = 'checkpoint'
    # Use glob to find matching paths
    # path is usually checkpoint_fold_1/checkpoint-<step number>
    # we are guaranteed to save only 1 checkpoint from training
    pattern = 'checkpoint-*'
    checkpoint_path = glob.glob(os.path.join(checkpoint_directory, pattern))[0]

    # run inference
    infer = Inference(checkpoint_path)
    infer.prepare_dataloader(test_df, batch_size=BATCH_SIZE)
    entity_prediction_list = infer.generate()

    # compare
    df_out = pd.DataFrame({
        'entity_pred': entity_prediction_list, 
    })
    df = pd.concat([test_df, df_out], axis=1)
    condition_correct = df['entity_name'] == df['entity_pred']
    pred_correct_proportion = sum(condition_correct)/len(df)

    # write output to file output.txt
    with open("output.txt", "w") as f:
        print(f'Accuracy: {pred_correct_proportion}', file=f)









# %%
if __name__ == '__main__':
    main()