from fastapi import FastAPI, HTTPException
import numpy as np
from PIL import Image
import os 
import json
from pydantic import BaseModel
from google.cloud import aiplatform
from typing import List

app = FastAPI()

PROJECT_ID = "projet-ia-448520"  
LOCATION = "us-central1"  
BUCKET_URI = f"gs://cours3bucket"
ID_ENDPOINT = "5825974565515296768"


class Payload(BaseModel):
    img : List[List[List[float]]]


aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
endpoint = aiplatform.Endpoint(ID_ENDPOINT)


@app.post('/call-endpoint/')
async def call_bedrock(payload: Payload):
    try:
        img = payload.img

        predictions = endpoint.predict(instances=[img])
        y_predicted = np.argmax(predictions.predictions, axis=1)
        print(y_predicted)
        return int(y_predicted[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Vertex: {e}") 