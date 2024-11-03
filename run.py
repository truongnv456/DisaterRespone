from flask import Flask, request, render_template
import pandas as pd
import plotly.express as px
import pickle
import os
import numpy as np
from sqlalchemy import create_engine

app = Flask(__name__)

# Define categories
categories = [
    'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
    'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter',
    'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
    'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers',
    'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
    'other_weather', 'direct_report'
]

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'pipeline_new.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load training data from SQLite database
def load_training_data(database_filepath):
    """Load training data from SQLite database."""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    return df

# Create a plot for the overview of the training dataset
def create_training_overview_plot(training_data):
    category_counts = training_data.iloc[:, 4:].sum().sort_values(ascending=False)
    df_overview = pd.DataFrame(category_counts).reset_index()
    df_overview.columns = ['Category', 'Count']
    fig_overview = px.bar(df_overview, x='Category', y='Count', title='Overview of Training Dataset')
    return fig_overview.to_html(full_html=False)

@app.route('/')
def home():
    # Load training data
    training_data = load_training_data('InsertDatabaseName.db')  # Path to your database

    # Create visualization
    training_overview_html = create_training_overview_plot(training_data)

    return render_template('index.html', training_overview_graph=training_overview_html, categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    message = request.form['message']
    
    # Make prediction
    prediction = model.predict([message])
    
    # Check the type of prediction
    if isinstance(prediction, np.ndarray):
        prediction = prediction.flatten() 

    # Create a prediction dictionary
    prediction_dict = {category: int(pred) for category, pred in zip(categories, prediction.tolist())}

    # Load training data again to create the overview plot
    training_data = load_training_data('InsertDatabaseName.db')  # Path to your database
    training_overview_html = create_training_overview_plot(training_data)

    return render_template('index.html', 
                           training_overview_graph=training_overview_html,
                           prediction=prediction_dict,
                           categories=categories)

if __name__ == '__main__':
    app.run(debug=True)
