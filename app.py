import pickle
import streamlit as st
import nltk
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk_data_path = 'D:/machinelearning/spamsmsdetection/nltk_data'
os.environ['NLTK_DATA'] = nltk_data_path
nltk.data.path.append(nltk_data_path)

# Preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    return ' '.join(filtered_tokens)

# Load the trained model and vectorizer from the .pkl files
with open('model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("Spam SMS Detector")
user_input = st.text_input("Enter a message to classify:")

if st.button("Predict"):
    # Preprocess the user input message
    cleaned_input = preprocess_text(user_input)
    
    # Convert the cleaned input to features using the fitted vectorizer
    new_message_features = vectorizer.transform([cleaned_input]).toarray()
    
    # Make the prediction using the trained model
    prediction = best_model.predict(new_message_features)
    
    # Display the result
    st.write("Prediction: Spam" if prediction[0] == 1 else "Prediction: Ham")
