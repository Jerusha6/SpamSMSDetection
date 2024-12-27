import streamlit as st
import pickle
import nltk
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time

nltk_data_path = 'D:/machinelearning/spamsmsdetection/nltk_data'
os.environ['NLTK_DATA'] = nltk_data_path

nltk.download('punkt')
nltk.download('stopwords')

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

# Load the trained model and vectorizer from .pkl files
with open('model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.markdown("""
    <style>
    .stButton>button {
        background-color: #ff6347;
        color: white;
        font-size: 20px;
    }
    .stTextInput>div>input {
        background-color: #f0f8ff;
        font-size: 18px;
        padding: 15px;
    }
    .stTextInput>div>label {
        font-size: 20px;
        color: #2f4f4f;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit title and description
st.title("🔍 Spam SMS Detector")
st.markdown("### Enter your SMS message below to check if it's **Spam** or **Ham**:")

# User input for SMS message
user_input = st.text_input("Enter a message:")

# Prediction logic
if st.button("🔮 Predict"):
    if user_input:
        # Show a spinner while prediction is being made
        with st.spinner("Making prediction..."):
            time.sleep(2)  # Simulate a slight delay for the prediction process
            cleaned_input = preprocess_text(user_input)
            
            # Convert the cleaned input to features using the fitted vectorizer
            new_message_features = vectorizer.transform([cleaned_input]).toarray()
            
            # Make the prediction using the trained model
            prediction = best_model.predict(new_message_features)
        
        st.success("Prediction complete!")
        
        # Display the result
        result = "Spam" if prediction[0] == 1 else "Ham"
        st.markdown(f"### Prediction: **{result}**")
        
    else:
        st.warning("Please enter a message to classify!")  # Warning if input is empty
