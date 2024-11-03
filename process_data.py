import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.
    
    Args:
    messages_filepath: str. Filepath for the messages dataset.
    categories_filepath: str. Filepath for the categories dataset.
    
    Returns:
    df: dataframe. Merged dataset.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    
    # Drop duplicates if any
    df = df.drop_duplicates()

    # Assert to check if there are any duplicates after merging
    assert len(df[df.duplicated()]) == 0, "Duplicates found in the merged DataFrame!"
    
    return df

def clean_data(df):
    """
    Clean the merged dataframe.
    
    Args:
    df: dataframe. Merged dataset.
    
    Returns:
    df: dataframe. Cleaned dataset.
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # Extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Convert category values to numbers (0, 1, 2) based on unique values
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])  # Get the value after the hyphen
        categories[column] = pd.to_numeric(categories[column], errors='coerce')  # Convert to numeric, handle errors
    
    # Drop the original categories column from df
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
       # Assert to check for duplicates in the cleaned DataFrame
    assert len(df[df.duplicated()]) == 0, "Duplicates found in the cleaned DataFrame!"
    
    return df

def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database.
    
    Args:
    df: dataframe. Cleaned dataset.
    database_filename: str. Filename for the output database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    """
    Main function to run the data processing steps.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        print("Please provide the file paths for messages, categories, and the database as command-line arguments.")
        return
    
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)
    
    print('Cleaning data...')
    df = clean_data(df)
    
    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)
    
    print('Cleaned data saved to database!')

if __name__ == '__main__':
    main()
