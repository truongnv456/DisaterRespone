from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Install model has been trained
model_path = os.path.join(os.path.dirname(__file__), 'model', 'pipeline.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Define categories class
categories = [
    'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
    'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter',
    'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
    'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers',
    'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
    'other_weather', 'direct_report'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive data from form
    message = request.form['message']
    
    # Prediction
    prediction = model.predict([message])
    
    # Check type of prediction
    if isinstance(prediction, np.ndarray):
        prediction = prediction.flatten() 

    # Create dictionnary
    prediction_dict = {category: int(pred) for category, pred in zip(categories, prediction.tolist())}

    return render_template('index.html', prediction=prediction_dict)

if __name__ == '__main__':
    app.run(debug=True)
