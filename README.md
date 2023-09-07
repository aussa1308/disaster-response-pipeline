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
