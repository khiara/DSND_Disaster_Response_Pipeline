# Disaster Response Pipeline Project
For the Udacity Data Science Nanodegree Program

## Table of Contents
1. Description
2. Preparation 
   - Installation
   - Dependencies
3. Program Execution 
   - ETL Pipeline
   - ML Pipeline
   - Web App
4. Screenshots
5. License and Acknowledgments


## 1.  Description
This project is part of the Udacity Data Science Nanodegree Program. The goal was to take actual social media messages sent in the aftermath of natural disasters, and create a Natural Language Processing model to classify those messages by category of response requested. The tuned model would then be deployed as a web app, into which a user could input their own message and get classification results showing the relevant categories. 
## 2. Preparation
### Installation
To clone this repository, use: <br />
     `git clone https://github.com/khiara/DSND_Disaster_Response_Pipeline.git`
     
### Dependencies
python (>= 3.6) <br />
**Libraries:** NumPy, Pandas, Scikit-learn, NLTK, SQLalchemy  <br />
**Web app and data visualization:** Flask, Plotly

## 3. Program Execution
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database: <br />
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves: <br />
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
2. Run the following command in the app's directory to run your web app: <br />
    `python run.py`

3. Open a browser window and go to http://0.0.0.0:3001/ or http://localhost:3001/.

## 4. Screenshots

## 5. Acknowledgments
- [Udacity](http://udacity.com) Data Science Nanodegree Program provided instruction, inspiration, and guidance.
- [Figure Eight](http://appen.com) provided the data.
