from flask import Flask, request, render_template
import plotly.express as px
import pickle
import numpy as np
import pandas as pd
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

# Function to create a plot from prediction data
def create_plot(prediction_dict):
    # Convert prediction data to a DataFrame
    df = pd.DataFrame(list(prediction_dict.items()), columns=['Category', 'Count'])
    
    # Create a bar chart
    fig = px.bar(df, x='Category', y='Count', title='Prediction Results')
    return fig.to_html(full_html=False)

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

    # Create a plot from prediction data
    graph_html = create_plot(prediction_dict)

    return render_template('index.html', prediction=prediction_dict, graph=graph_html)

if __name__ == '__main__':
    app.run(debug=True)
