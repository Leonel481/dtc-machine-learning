import pickle
from fastapi import FastAPI
from typing import Annotated
from pydantic import BaseModel, StringConstraints, Field
import uvicorn


# Load the pre-trained model from a file
# Testing local use pipeline_v1.bin
with open('pipeline_v2.bin', 'rb') as model_file:
        model = pickle.load(model_file)

def predict_record(record):
    prediction = model.predict([record])
    return float(prediction[0])

class Lead(BaseModel):
    lead_source: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    number_of_courses_viewed: Annotated[int, Field(ge=0)]
    annual_income: Annotated[float, Field(ge=0)]

class PredictionRequest(BaseModel):
    predict: Annotated[float, Field(ge=0)]

app = FastAPI(title="record-prediction-api")

@app.post("/predict")
def predict(record: Lead) -> PredictionRequest:

    predict = predict_record(record.dict())

    return {"predict": predict}


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)