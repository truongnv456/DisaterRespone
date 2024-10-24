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

# Tải dữ liệu từ cơ sở dữ liệu
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table('InsertTableName', engine)

X = df['message']
y = df.iloc[:, 4:]

def self_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in words if word not in stop_words]
    return lemmas

X = X.apply(self_tokenize)
X = X.apply(lambda x: ' '.join(x))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer()),
        ('count', CountVectorizer())
    ])), # type: ignore
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

num_labels = y_test.values.shape[1]

for i in range(num_labels):
    report = classification_report(y_test.values[:, i], y_pred[:, i], output_dict=True) # type: ignore
    if isinstance(report, dict):
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"Label: {label}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1-Score: {metrics['f1-score']:.4f}")
    else:
        print("Error: classification_report did not return a dictionary.")

with open('pipeline_new.pkl', 'wb') as file:
    pickle.dump(pipeline, file)
