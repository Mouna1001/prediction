# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 23:57:36 2021

@author: ASUS
"""
import warnings
warnings.filterwarnings('ignore')
import nest_asyncio
nest_asyncio.apply()
# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 2. Create the app object
app = FastAPI()

def read_pickeld(pickle_name):
    pickle_in = open(pickle_name, "rb")
    pickled_file = pickle.load(pickle_in)
    return pickled_file

rfc = read_pickeld("rfc_model.pkl")
barrage_encoder = read_pickeld('barrage_encoder.pkl')
month_encoder = read_pickeld('month_encoder.pkl')
day_encoder = read_pickeld('day_encoder.pkl')

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}


# 4. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence

# PS: You can create a html template to collect this data from the browser, but for now.
# Create dummy data

class Item(BaseModel):
    Name: str
    Reserve: int
    day_name: str
    month_name: str
    Year: int
    
# data = {
#     'Name': 'Oued El Makhazine',
#     'Reserve': 630,
#     'day_name': 'Friday',
#     'month_name': 'July',
#     'Year': 2025,
# }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
@app.post('/predict')
def predict_fill_rate(data: Item):
    data = data.dict()
    print(data)
    data = pd.DataFrame(data, index=range(len(data)))
    data['Name'] = barrage_encoder.transform(data['Name'])
    data['month_name'] = month_encoder.transform(data['month_name'])
    data['day_name'] = day_encoder.transform(data['day_name'])
    prediction = rfc.predict(data)[0]
    message = 'Prediction du taux de remplissage = {}'.format(prediction)
    return {'message': message}


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload

