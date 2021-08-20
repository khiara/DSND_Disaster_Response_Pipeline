import sys

# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load two csv files containing disaster response data and merge into a single dataframe variable
    INPUTS:
    messages_filepath, categories_filepath -- filepaths to csv files
    OUTPUT:
    df -- merged DataFrame combining messages and categories datasets
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how = 'inner', on = ['id'])
    
    return df


def clean_data(df):
    '''
    Split the values in the categories column on the ; character so that each value becomes a separate column. 
    Use the first row of categories dataframe to create column names for the categories data, then rename 
    columns of categories with new column names.
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
    # convert column from string to numeric
        categories[column]= pd.to_numeric(categories[column])
    
    # replace '2' values in 'related' column with '1'
    categories['related'] = categories['related'].replace(2,1)
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join = 'inner', sort = 'False')
    
    # drop duplicates
    df.drop_duplicates(keep = 'first', inplace = True)
    
    return df


def save_data(df, database_filename):
    '''
    Save the clean dataset into an sqlite database
    INPUTS:
    df, database_filename -- name of SQL database file
    OUTPUT:
    none
    '''
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_response_data', engine, index=False, if_exists='replace')  


def main():
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