# %%
# imports
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# %%
# generate label and id maps
data_path = '../../data/process/train.csv'
full_df = pd.read_csv(data_path)
entity_list = sorted(list((set(full_df['entity_name']))))

id2label = {}
label2id = {}
for idx, val in enumerate(entity_list):
    id2label[idx] = val
    label2id[val] = idx

# %%
# create a list of corresponding id's
train_label = full_df['entity_name'].map(label2id)

# %%
##################################################
# helper functions


class Retriever:
    def __init__(self, input_texts, model_checkpoint):
        # we need to generate the embedding from list of input strings
        self.embeddings = []
        self.inputs = input_texts
        model_checkpoint = model_checkpoint 
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt", clean_up_tokenization_spaces=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        model.to(self.device)
        self.model = model.eval()


    def make_embedding(self, batch_size=64):
        all_embeddings = self.embeddings
        input_texts = self.inputs

        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i+batch_size]
            # Tokenize the input text
            # since we tokenize per batch, we don't need a data collator
            # also, the in-order nature also means we don't need a dataloader
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)


            # Pass the input through the encoder and retrieve the embeddings
            with torch.no_grad():
                encoder_outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                # get last layer
                embeddings = encoder_outputs.hidden_states[-1]
                # get cls token embedding
                cls_embeddings = embeddings[:, 0, :]  # Shape: (batch_size, hidden_size)
                all_embeddings.append(cls_embeddings)
        
        # remove the batch list and makes a single large tensor, dim=0 increases row-wise
        all_embeddings = torch.cat(all_embeddings, dim=0)

        self.embeddings = all_embeddings


def cosine_similarity_chunked(batch1, batch2, chunk_size=1024):
    device = 'cuda'
    batch1_size = batch1.size(0)
    batch2_size = batch2.size(0)
    batch2.to(device)
    
    # Prepare an empty tensor to store results
    cos_sim = torch.empty(batch1_size, batch2_size, device=device)

    # Process batch1 in chunks
    for i in range(0, batch1_size, chunk_size):
        batch1_chunk = batch1[i:i + chunk_size]  # Get chunk of batch1
        
        batch1_chunk.to(device)
        # Expand batch1 chunk and entire batch2 for comparison
        # batch1_chunk_exp = batch1_chunk.unsqueeze(1)  # Shape: (chunk_size, 1, seq_len)
        # batch2_exp = batch2.unsqueeze(0)  # Shape: (1, batch2_size, seq_len)
        batch2_norms = batch2.norm(dim=1, keepdim=True)

        # Compute cosine similarity by matrix multiplication and normalizing
        sim_chunk = torch.mm(batch1_chunk, batch2.T) / (batch1_chunk.norm(dim=1, keepdim=True) * batch2_norms.T + 1e-8)
        
        # Store the results in the appropriate part of the final tensor
        cos_sim[i:i + chunk_size] = sim_chunk
    
    return cos_sim

# the following function takes in a full cos_sim_matrix
# condition_source: boolean selectors of the source embedding
# condition_target: boolean selectors of the target embedding
def find_closest(cos_sim_matrix, condition_source, condition_target):
    # subset_matrix = cos_sim_matrix[condition_source]
    # except we are subsetting 2D matrix (row, column)
    subset_matrix = cos_sim_matrix[np.ix_(condition_source, condition_target)]
    # we select top k here
    # Get the indices of the top k maximum values along axis 1
    top_k = 3
    top_k_indices = np.argsort(subset_matrix, axis=1)[:, -top_k:]  # Get indices of top k values
    # note that top_k_indices is a nested list because of the 2d nature of the matrix
    # the result is flipped
    top_k_indices[0] = top_k_indices[0][::-1]
    
    # Get the values of the top 5 maximum scores
    top_k_values = np.take_along_axis(subset_matrix, top_k_indices, axis=1)
    

    return top_k_indices, top_k_values




class Embedder():
    input_df: pd.DataFrame
    fold: int

    def __init__(self, input_df):
        self.input_df = input_df


    def make_embedding(self, checkpoint_path):

        def generate_input_list(df):
            input_list = []
            for _, row in df.iterrows():
                desc = row['mention']
                input_list.append(desc)
            return input_list

        # prepare reference embed
        train_data = list(generate_input_list(self.input_df))
        # Define the directory and the pattern
        retriever_train = Retriever(train_data, checkpoint_path)
        retriever_train.make_embedding(batch_size=64)
        return retriever_train.embeddings.to('cpu')


# %%
data_path = f'../../data/process/test.csv'
test_df = pd.read_csv(data_path)

data_path = f"../../data/process/train.csv"
train_df = pd.read_csv(data_path, skipinitialspace=True)

checkpoint_directory = "checkpoint"
# Use glob to find matching paths
# path is usually checkpoint_fold_1/checkpoint-<step number>
# we are guaranteed to save only 1 checkpoint from training
pattern = 'checkpoint-*'
checkpoint_path = glob.glob(os.path.join(checkpoint_directory, pattern))[0]

train_embedder = Embedder(input_df=train_df)
train_embeds = train_embedder.make_embedding(checkpoint_path)

test_embedder = Embedder(input_df=test_df)
test_embeds = test_embedder.make_embedding(checkpoint_path)

# test embeds are inputs since we are looking back at train data
cos_sim_matrix = cosine_similarity_chunked(test_embeds, train_embeds, chunk_size=1024).cpu().numpy()

# take the top 1 closest match
top_k = 1
top_k_indices = np.argsort(cos_sim_matrix, axis=1)[:, -top_k:]  # Get indices of top k values
top_k_indices = top_k_indices.squeeze(-1) # back to 1dim array
preds = [train_label[i] for i in top_k_indices]



y_true = test_df['entity_name'].map(label2id)
y_pred = preds
# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)

# %%
with open("similarity_output.txt", "w") as f:
    # Print the result
    print(f'Accuracy: {accuracy:.5f}', file=f)