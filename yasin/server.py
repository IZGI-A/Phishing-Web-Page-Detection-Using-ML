import os

import joblib
import numpy as np
from flask import Flask, render_template, request

from prepare_embedding import extract_and_copy_text, translate_and_append_into, embed

app = Flask(__name__)

# Load your XGBoost model
model = joblib.load('model/xlm-roberta_xgb_based_model.pkl')
model_name = "xlm-roberta"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    prediction_result = "Phishing"  # or it will be "Legitimate"
    if 'htmlFile' not in request.files:
        return "No file part"

    file = request.files['htmlFile']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file to a temporary location

    file_path = 'test/' + file.filename
    translated_file_path = 'test_Translated/' + file.filename
    file.save(file_path)

    # START of the business logic here
    # Perform prediction using the file_path with your XGBoost model
    # Replace the following line with your actual prediction logic

    global model_name

    extract_and_copy_text(file.filename, "test", "test")

    translate_list = []
    translate_and_append_into(file_name=file.filename, source_path="test", append_into=translate_list,
                              model_name=model_name, cache=False)

    embedded = np.zeros(shape=(1, 768))
    embedded[0] = embed(translate_list[0], model_name)

    pred = model.predict(embedded)[0]

    is_phishing = (pred == 1)
    if not is_phishing:
        prediction_result = "Legitimate"


    # END of the business logic here
    return f"{file_path} is {prediction_result}"


if __name__ == '__main__':
    app.run(debug=True, port=5050)
