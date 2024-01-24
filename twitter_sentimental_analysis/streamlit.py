import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def main():
    # Title
    st.title("Twitter Sentimental Analysis")
    st.write("About 5 million tweets are tweeted everyday, some of them positive while some negative. Trained on a dataset of 16 million tweets, this model will allow you to predict whether a tweet is positive or negative.")

#sideBar with 2 options:

    selected_option = st.sidebar.radio("Select Option", [ "Home", "Predict"])
    if selected_option == "Predict":
        predict_section()
    elif selected_option == "Home":
        home_section()
        
#home section
def home_section():
    
    # Dropdown to select either a dataset or a model
    selection = st.selectbox("Select what you want to know about", ["Dataset", "Model"])


    if selection == "Dataset":
        # Display the class distribution for 1000 random tweets from the dataset
        st.header('Dataset info')
        st.write('The dataset used here was downloaded from Kaggle. The dataset is called Sentiment140, wherein 16 million tweets with their timestamp, user and ids are mentioned')

    elif selection== "Model":
        st.header('Model info')
        st.write('I have used a linear regression model to categorize the data into positive or negative tweets. Linear regression is a supervised Machine Learning technique that can be used in case of categorical data tasks.')
    

    
#prediction section
def predict_section():
    st.subheader("Prediction Section")
    
    user_input = st.text_area("Enter text for sentiment analysis:") 
    
    if st.button("Analyze Sentiment"):
        if user_input:
            perform_sentiment_analysis(user_input)
        else:
            st.warning("Please enter text for analysis.")

def perform_sentiment_analysis(text):
    
    model = pickle.load(open('trained_modelSentimental.sav', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    
    user_input_tfidf = vectorizer.transform([text])
    prediction = model.predict(user_input_tfidf)
    if prediction[0] == 1:
        st.write('This is a positive tweet')
    else:
        st.write('This is a negative tweet') 
   

if __name__ == "__main__":
    main()
