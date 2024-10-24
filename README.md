# Disater Response 

## Introduction
Currently, due to abnormal weather changes, natural disasters are increasing. Therefore, developing a system for quickly responding to and coordinating relief activities is extremely necessary. “Disaster Response” project is built to address this purpose.

The main goal of the project is to classify the messages received in disaster situations into different categories such as requests for help, offers of assistance, information related to medical needs, financial aid, or missing persons.

The project uses data provided in [[Figure Eight]](https://www.appen.com/) and builds machine learning  model to classify the message received in disater situations.

## Installation
### 1. Clone the repository: 
    git clone https://github.com/truongnv456/DisaterRespone/tree/master
### 2. Navigate to the project directory and install requirements: 
    cd DiasterResponse
    pip install -r requirement.txt
### 3. Usage
To start the application, run:

    python run.py

Then go to http://127.0.0.1:5000/

## Files

- `README.md`: Manual file, providing an overview of the document and how to use it.
- `requirements.txt`: The file contains a list of libraries required to run the notebook.
- `ETL Pipeline Preparation.ipynb`: This jupyter Notebook file containing source code ELP pipeline, prepare data for MLP.
- `ML Pipeline Preparation.ipynb`: This jupyter Notebook file containing source code MLP pipeline to build a pipeline.
- `run.py`: Launch the Flask web app used to classify disaster messages

## References

- [[Figure Eight]](https://www.appen.com/): About Disater Response dataset to train.
- [[Udacity]](https://www.udacity.com): About learning this lesson.

## Screen shoots
1. Home page: you can enter the message in here

![HomePage](screenshoot/home.png)

2. Enter message and click 'Predict' button

![Predict](screenshoot/message.png)

3. After you click, you can see the message is in categories, which message is classify to

![After](screenshoot/classify.png)


