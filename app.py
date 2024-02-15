from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load CSV data
data = pd.read_csv("aakash data - Sheet3.csv")

# Create Flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

categorical_cols = ['state', 'mosquito']
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Retrieve form data
    form_data = request.form

    # Convert form data into pandas DataFrame
    input_data = {
        'cases': [float(form_data['feature1'])],
        'rainfall in mm': [float(form_data['feature2'])],
        'temperature': [float(form_data['feature3'])],
        'avg relative humidity': [float(form_data['feature4'])],
        'state': [form_data['feature5']],
        'mosquito': [form_data['feature6']]
    }
    df = pd.DataFrame(input_data)

    # Preprocess input data
    numeric_features = ['cases', 'rainfall in mm', 'temperature', 'avg relative humidity']
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    # Encode categorical features
    encoded_data = encoder.transform(df[categorical_cols])
    df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols)

    # Concatenate numeric and encoded categorical features
    df_combined = pd.concat([df[numeric_features], df_encoded], axis=1)

    # Make prediction
    prediction = model.predict(df_combined)
    probability=model.predict_proba(df_combined)
    return render_template("index.html", prediction_text="The prediction is {} and the probabilities of three diseases are {}".format(prediction, probability))

if __name__ == "__main__":
    app.run(debug=True)
