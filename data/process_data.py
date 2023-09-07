"""
Data Processing for Disaster Response

This script is responsible for loading, cleaning, and saving message and category
data for a disaster response dataset. It takes CSV files as inputs and outputs a 
cleaned SQLite database.

Functions:
    load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
        Load message and category data from CSV files and merge them.
        
    clean_data(df: pd.DataFrame) -> pd.DataFrame:
        Clean the merged DataFrame, splitting categories into separate columns and converting values to binary.

    save_data(df: pd.DataFrame, database_filename: str):
        Save the cleaned DataFrame to an SQLite database.
        
    main():
        The main function that orchestrates the loading, cleaning, and saving of data.
"""

# import libraries

import sys
import os
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load message and category data from CSV files and merge them.
    
    Args:
    messages_filepath (str): File path of the messages dataset.
    categories_filepath (str): File path of the categories dataset.
    
    Returns:
    pd.DataFrame: Merged dataset containing messages and categories.
    """
    messages   = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    """
    Clean the merged DataFrame by splitting categories into separate columns
    and converting category values to binary (0 or 1).
    
    Args:
    df (pd.DataFrame): Original merged DataFrame containing messages and categories.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with messages and converted categories.
    """
    categories        = df['categories'].str.split(';', expand=True)
    row               = categories.iloc[0]
    category_colnames = [name.split('-')[0] for name in row]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
        
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1, join='inner')
        
    # drop multi-label
    df = df[df["related"] != 2]
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to an SQLite database.
    
    Args:
    df (pd.DataFrame): Cleaned DataFrame containing messages and categories.
    database_filename (str): File path of the SQLite database.
    
    Returns:
    None
    """    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)
    

def main():
    """
    Main function to orchestrate the data processing.
    
    This function executes the following tasks:
    - Load the message and category data from CSV files.
    - Clean the merged DataFrame.
    - Save the cleaned DataFrame to an SQLite database.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
