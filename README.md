# disaster-response-pipeline

## Overview
This endeavor is a segment of the Data Science Learning Track offered by Udacity, in association with Figure Eight. It aims at creating an automated system that filters and categorizes disaster-related messages in real-time. This enables efficient and swift action by directing these messages to appropriate relief agencies.

A web-based interface is included to help emergency responders input received messages and obtain the associated classifications.

## Repository Contents

### App Folder
- `run.py`: Python script to initiate the web application.
- **Templates Folder**: Includes HTML templates (`go.html & master.html`) essential for the web application.

### Data Folder
- `disaster_messages.csv`: Texts exchanged during various emergencies (Source: Figure Eight).
- `disaster_categories.csv`: Classification types for the messages.
- `process_data.py`: The script responsible for ETL (Extract, Transform, Load) operations.
- `DisasterResponse.db`: Database containing the cleaned data.

### Models Folder
- `train_classifier.py`: Python script for machine learning operations.
- `classifier.pkl`: Serialized machine learning model.

## How to Run

To execute the program, you'll need to perform several tasks in the project folder:

- Initialize the database and clean the data by running:
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

- Load data from the database, train your machine learning model, and save it as a pickle file with this command:
python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl

- To launch the web application, navigate to the app folder and type:
python run.py

Once the web app is running, you can access it via http://0.0.0.0:3001/
