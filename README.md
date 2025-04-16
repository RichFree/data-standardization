# Data Standardization

This repository uses language models (LM) to perform data standardization on
variable item descriptions to give canonical item names.

The dataset `esAppMod`, which is a collection of enterprise application terms,
is sourced from
[tackle-container-advisor](https://github.com/konveyor/tackle-container-advisor).

The application has both a `frontend` and a `backend`. The `backend` provides
the api interface to access the model and is implemented as a FastAPI server.
The `frontend` provides the browser UI to query the API server.

## Machine Learning

### Dependencies
To setup, you can install from the `requirements.txt`.

```bash
uv venv --python 3.12 torch_env
source torch_env/bin/activate
uv pip install -r requirements.txt
```

But this `requirements.txt` contains my working environment packages, which
might include more packages than needed.

The main packages used were: pytorch, transformers, datasets, evaluate, numpy,
pandas.

The FastAPI package is also included in the requirements.

### Data Preparation

```python
python data/process/process_data.py
```

This will produce the files `{entity, train, test}.csv`, which are necessary for
training and testing.

### Training and Test

Currently the code employs 2 possible methods: classification, and sequence-to-sequence.

To run the classification section:

```python
cd train/classification
python train.py
python predict.py
cat output.txt
```

To run the seq2seq section:

```python
cd train/seq2seq
python train.py
python predict.py
cat output.txt
```

## Web Application

This section is split into the `backend` and `frontend` sections.

This application only uses the classification model for simplicity.

### Backend

This is a FastAPI backend that serves the trained model from earlier via a
Rest API.

To run:

```python
cd backend
fastapi dev predict.py
# OR
fastapi run predict.py
```

This opens a localhost link at port 8000.

### Frontend

This is a simple react frontend that allows queries to the earlier
Rest API. 

To run:

```python
cd frontend
yarn dev
```