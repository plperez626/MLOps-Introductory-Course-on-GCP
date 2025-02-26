import boto3
from locust import HttpUser, between, task
import json 
import numpy as np
from PIL import Image
import os 

def load_data():
    IMAGE_DIRECTORY = "cifar_test_images"

    image_files = [file for file in os.listdir(IMAGE_DIRECTORY) if file.endswith(".jpg")]
    image_data = [
        np.asarray(Image.open(os.path.join(IMAGE_DIRECTORY, file))) for file in image_files
    ]
    x_test = [(image / 255.0).astype(np.float32).tolist() for image in image_data]
    y_test = [int(file.split("_")[1]) for file in image_files]
    return x_test, y_test

x_test, y_test = load_data()
class APIUser(HttpUser):
    wait_time = between(1, 3)  

    @task
    def call_endpoint(self):
        single_image = x_test[1]
        
        
        payload = {"img": json.loads(json.dumps(single_image))}
        
      
        response = self.client.post("/call-endpoint/", json=payload)
        
       
        if response.status_code == 200:
            print(response.json())
        else:
            print(f"Erreur {response.status_code}: {response.text}")