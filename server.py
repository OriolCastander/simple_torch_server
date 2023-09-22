from flask import Flask, request, jsonify

import numpy as np
import torch

from model.model import Model
from model.normalizer import PolynomicNormalizer
import joblib


server = Flask(__name__)

normalizer: PolynomicNormalizer = joblib.load("mockData/normalizer.pkl")
model = Model.load("model/model.pt", 2, [10,10,10], 1)


@server.route('/')
def home():
    return "Home"

@server.route('/predict')
def predict():

    ##request.query_string comes as b'param1=value1&param2=value2'...

    try:
        paramList = str(request.query_string)[2:-1].split("&")
        params = {param.split("=")[0]: int(param.split("=")[1]) for param in paramList}

        assert all(key in params for key in ["day", "month", "time"])
    except:
        return "Params not correctly specified: Needs /predict?day=int&month=int&time=int, where time is between 0 and 24"
    

    dataset = np.array([[float(params["month"] * 30.5 + params["day"]), float(params["time"])]])
    normalizedDataset = normalizer.normalizeDataset(dataset)

    normalizedResponse = model.predict(torch.tensor(normalizedDataset, dtype=torch.float), None)
    response = normalizer.denormalizeColumns([2], normalizedResponse)

    return jsonify({"predictedTemp": float(response[0][0])})