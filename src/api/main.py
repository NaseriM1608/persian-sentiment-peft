from fastapi import FastAPI
from pydantic import BaseModel
from src.api.inference import predict

app = FastAPI()

class ReviewRequest(BaseModel):
    text: str


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    return {"text": predict(request.text)}