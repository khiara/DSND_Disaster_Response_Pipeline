import sys

# import libraries
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import classification_report, accuracy_score

import pickle


def load_data(database_filepath):
    '''
    Load the SQL database and create the features from the message data and labels from the category data
    INPUT:
    database_filepath -- path to SQL database
    OUTPUT:
    X -- features DataFrame
    Y -- labels DataFrame
    category_names
    '''
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('disaster_response_data', engine)
    
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
  
    return X, y, category_names


def tokenize(text):
    '''
    Process the text to prepare it for modeling -- remove html, irelevant words, punctuation, and capital letters, then 
    split words into individual tokens.
    '''
    # find url tags and replace with 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # remove punctuation and convert text to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    # lemmatize text and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Create a new feature indicating whether or not a sentence starts with a verb
    '''
    
    def starting_verb(self, text):
        # tokenize sentences
        sent_list = nltk.sent_tokenize(text)
        
        for sentence in sent_list:
            # tokenize words and tag for parts of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            # get first word and part of speech and determine if it is a verb or RT
            if len(pos_tags) > 1:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            else:
                return False
        return False
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
    
def build_model():
    '''
    Build a machine learning pipeline to create a classifier model,and use GridSearch to find best hyperparameters.
    '''
    # build the pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor()),
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
    'clf__estimator__n_estimators': [50, 100, 200],
    'clf__estimator__learning_rate':[0.001, 0.01, 0.1, 0.2, 0.5]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose = 3)
    
    return model


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Test how well the pipeline performs, checking how well it predicts the label given the message
    INPUTS:
    model -- the pipeline
    X_test -- test set message values
    y_test -- test set label values
    category_names
    OUTPUT:
    classification_report and flattened_accuracy_score
    '''
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, target_names = category_names)
    accuracy = accuracy_score(y_test.values.reshape(-1,1), y_pred.reshape(-1,1))
    
    print('Accuracy overall {0:.2f}% \n'.format(accuracy*100))
    print(report)
    
    return None


def save_model(model, model_filepath):
    '''
    Save and export the model as a pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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