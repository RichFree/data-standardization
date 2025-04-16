# %%
import json
import pandas as pd
from pathlib import Path

# %%
# setup
BASE_DIR = Path(__file__).resolve().parent

# part 1: process the entities
##########################################
# %%

# Load the JSON file
data_path = BASE_DIR / '../esAppMod/tca_entities.json'
with open(data_path, 'r') as file:
    data = json.load(file)

# Initialize an empty list to store the rows
rows = []

# %%
# Loop through all entities in the JSON
for entity in data["data"].items():
    entity_data = entity[1]
    entity_id = entity_data['entity_id']
    entity_name = entity_data['entity_name']
    entity_type_id = entity_data['entity_type_id']
    entity_type_name = entity_data['entity_type_name']
    
    # Add each mention and its entity_id to the rows list
    rows.append(
        {
        'id': entity_id,
        'name': entity_name,
        'type_id': entity_type_id,
        'type_name': entity_type_name
        })

# Create a DataFrame from the rows
df = pd.DataFrame(rows)

# %%
df.to_csv(BASE_DIR / 'entity.csv', index=False)



# part 2: process the training data
##########################################
# %%
# import entity information

# %%
data_path = BASE_DIR / 'entity.csv'
entity_df = pd.read_csv(data_path, skipinitialspace=True)
id2label = {}
for _, row in entity_df.iterrows():
    id2label[row['id']] = row['name']


# Load the JSON file
data_path = BASE_DIR / '../esAppMod/train.json'
with open(data_path, 'r') as file:
    data = json.load(file)

# Initialize an empty list to store the rows
rows = []

# Loop through all entities in the JSON
for entity_key, entity_data in data["data"].items():
    mentions = entity_data["mentions"]
    entity_id = entity_data["entity_id"]
    entity_name = id2label[entity_id]
    
    # Add each mention and its entity_id to the rows list
    for mention in mentions:
        rows.append(
            {
                "mention": mention,
                "entity_id": entity_id,
                "entity_name": entity_name
            })

# Create a DataFrame from the rows
train_df = pd.DataFrame(rows)

train_class_set = set(train_df['entity_id'].to_list())

# %%
train_df.to_csv(BASE_DIR / 'train.csv', index=False)

# part 3: process the test data
##########################################
# %%
# Load the JSON file
data_path = BASE_DIR / '../esAppMod/infer.json'
with open(data_path, 'r') as file:
    data = json.load(file)

# Initialize an empty list to store the rows
rows = []

# Loop through all entities in the JSON
for entity_key, entity_data in data["data"].items():
    mention = entity_data["mention"]
    entity_id = entity_data["entity_id"]
    entity_name = id2label[entity_id]
    
    # Add each mention and its entity_id to the rows list
    rows.append(
        {
            "mention": mention,
            "entity_id": entity_id,
            "entity_name": entity_name
        })



# Create a DataFrame from the rows
test_df = pd.DataFrame(rows)

test_class_set = (set(test_df['entity_id'].to_list()))

# %%
test_df.to_csv(BASE_DIR / 'test.csv', index=False)

# %%
# this shows that the training data can be found in the train set
# print(test_class_set - train_class_set )

# %%
