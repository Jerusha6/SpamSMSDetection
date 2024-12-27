import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Example: Streamlit input section
st.title("Spam SMS Detector")
user_input = st.text_input("Enter a message to classify:")

# When the user presses the button
if st.button("Predict"):
    # Preprocess the user input message
    cleaned_input = preprocess_text(user_input)

    # Convert the cleaned input to features using the trained vectorizer
    new_message_features = vectorizer.transform([cleaned_input]).toarray()

    # Make prediction with the trained model
    prediction = best_model.predict(new_message_features)
    
    # Show the prediction result
    st.write("Prediction: Spam" if prediction[0] == 1 else "Prediction: Ham")
