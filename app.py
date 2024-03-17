# Random Forrest model deployment

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json

# need sklearn 1.3.1

# Create flask app
flask_app = Flask(__name__)
#model = pickle.load(open(r"/Users/teja/Desktop/HeroVired AI ML DS/0_Capstone_Project/model_RF_v2_deployment/model_RF_v2.pkl", "rb"))
model = pickle.load(open(r"./model_RF_v2.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    text_features = list(request.form.values())  # Convert values to a list
    preprocessed_features = [text.lower() for text in text_features]  # Convert text to lowercase
    prediction = model.predict(preprocessed_features)  # Predict using preprocessed features
    
    with open('./mappings.json') as f:
        mappings = json.load(f)
    
    description = list(mappings['all'].keys())[list(mappings['all'].values()).index(prediction)]
    
    return render_template("index.html", prediction_text = "The Complaint Category is {0} Description: {1}".format(prediction, description))

if __name__ == "__main__":
    flask_app.run(debug=True)
