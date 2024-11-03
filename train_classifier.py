import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import re
import pickle
import sys

def load_data(database_filepath, table_name):
    """Load data from SQLite database."""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name, engine)
    X = df['message']
    y = df.iloc[:, 4:]
    return X, y

def tokenize(text):
    """Tokenize and lemmatize the input text."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in words if word not in stop_words]
    return lemmas

def build_pipeline():
    """Build a machine learning pipeline."""
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
            ('count', CountVectorizer(tokenizer=tokenize))
        ])), # type: ignore
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline

def build_model():
    """Build a machine learning pipeline with GridSearchCV."""
    pipeline = build_pipeline()
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__max_depth': [None, 10, 20],
        'clf__estimator__min_samples_split': [2, 5]
    }
    
    # Create GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=3)
    return grid_search

def evaluate_model(y_test, y_pred):
    """Evaluate the model and print classification report."""
    num_labels = y_test.values.shape[1]
    for i in range(num_labels):
        report = classification_report(y_test.values[:, i], y_pred[:, i], output_dict=True)
        if isinstance(report, dict):
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    print(f"Label: {label}")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall: {metrics['recall']:.4f}")
                    print(f"F1-Score: {metrics['f1-score']:.4f}")
        else:
            print("Error: classification_report did not return a dictionary.")

def save_model(pipeline, model_filepath):
    """Save the trained model to a pickle file."""
    with open(model_filepath, 'wb') as file:
        pickle.dump(pipeline, file)

def main():
    """Main function to run the pipeline."""
    if len(sys.argv) == 3:
        database_filepath = sys.argv[1]
        model_filepath = sys.argv[2]
        table_name = 'InsertTableName'  # You can also pass this as an argument if needed
    else:
        print("Please provide the database filepath and model filepath as command-line arguments.")
        return
    
    # Load data
    X, y = load_data(database_filepath, table_name)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the pipeline with GridSearch
    model = build_model()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

    # Save the model
    save_model(model, model_filepath)

if __name__ == '__main__':
    main()
