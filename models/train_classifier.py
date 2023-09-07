"""
Disaster Response Pipeline

This script is designed to read data from a SQLite database, tokenize the data,
build a machine learning model for text classification, evaluate the model, and
save the model to a file.

Functions:
    load_data(database_filepath: str) -> Tuple[pd.Series, pd.DataFrame, pd.Index]:
        Load the dataset from SQLite and return X, y and category names.

    tokenize(text: str, url_place_holder_string: str = "urlplaceholder") -> List[str]:
        Tokenize the text and return a list of tokens.

    build_model() -> Pipeline:
        Build the text classification model pipeline.

    evaluate_model(model: Pipeline, X_test: pd.Series, Y_test: pd.DataFrame, category_names: pd.Index):
        Evaluate the model performance on test data and print the classification report.

    save_model(model: Pipeline, model_filepath: str):
        Save the model to a file.

    main():
        Main function to execute the script tasks.

"""

# import libraries
import sys
import os
import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine
from typing import List

# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, HashingVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data(database_filepath):
    """
    Load the dataset from SQLite and return X, y and category names.

    Parameters:
    database_filepath (str): Filepath of SQLite database.

    Returns:
    pd.Series: Messages (X)
    pd.DataFrame: Categories (y)
    pd.Index: Category names
    """    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response',engine)
    
    df = df.loc[df['related'] != 2]
    
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    
    return X, y, category_names
    
def tokenize(text: str, url_place_holder_string: str = "urlplaceholder") -> List[str]:
    """
    Tokenize the text and return a list of tokens.

    Parameters:
    text (str): Text message
    url_place_holder_string (str): Placeholder for URL in text (default "urlplaceholder")

    Returns:
    List[str]: List of tokens
    """
    url_regex     = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    modified_text = text
    for detected_url in detected_urls:
        modified_text = modified_text.replace(detected_url, url_place_holder_string)
    
    tokens = nltk.word_tokenize(modified_text)
    lemmatizer = nltk.WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    return clean_tokens


def build_model():
    """
    Build the text classification model pipeline.

    Returns:
    Pipeline: Text classification model pipeline
    """
    model = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('hashing_vectorizer', HashingVectorizer(tokenizer=tokenize, n_features=10000, non_negative=True)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
        ])),
        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5)

    return grid_search


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model performance on test data and print the classification report.

    Parameters:
    model (Pipeline): Trained model
    X_test (pd.Series): Test messages
    Y_test (pd.DataFrame): True categories for test messages
    category_names (pd.Index): Names of categories

    Returns:
    None
    """
    
    y_predict_test  = model.predict(X_test)

    print(classification_report(Y_test.values, y_predict_test, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save the model to a file.

    Parameters:
    model (Pipeline): Trained model
    model_filepath (str): Filepath to save the model

    Returns:
    None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    Main function to execute the script tasks.
    
    Reads command-line arguments for database and model filepaths, loads data,
    builds and trains the model, evaluates the model, and saves it to a file.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
