from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from model import test

from fastapi.middleware.cors import CORSMiddleware

# Dummy ML model
def run_model(inputs: List[str]) -> List[str]:
    return test(inputs)

app = FastAPI()

# handle cross-origin request
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or set your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class Input(BaseModel):
    texts: List[str]

class Output(BaseModel):
    results: List[str]

@app.post("/predict", response_model=Output)
def predict(input: Input):
    return Output(results=run_model(input.texts))

