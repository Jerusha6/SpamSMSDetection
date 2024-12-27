# Spam SMS Detection App

This application uses machine learning techniques to classify SMS messages as either "Spam" or "Ham" (not spam). It uses a **Random Forest Classifier** model trained on a dataset of SMS messages. The model is integrated with a **Streamlit** web interface, allowing users to input SMS text and receive real-time predictions on whether the message is spam or not.

This project aims to demonstrate a practical application of machine learning in text classification and showcase how simple web applications can be built using Streamlit.


## Features:

- **Real-time SMS Classification**: The app allows users to input an SMS and immediately classify it as "Spam" or "Ham".
- **User-Friendly Interface**: Built with Streamlit for a clean, interactive UI.
- **Accuracy Display**: After the prediction, the app displays whether the message is **Spam** or **Ham**.
- **Machine Learning Model**: The backend uses a **Random Forest Classifier** to make the predictions.


## How to Use

1. Open the deployed app via the following link: [Spam SMS Detection](https://spamsmsdetection-pslhvrqrykjgvrs3b7qev2.streamlit.app/).
2. On the main page, you’ll see a text input field where you can enter an SMS message.
3. Type in any SMS message (real or test message).
4. Click the "Predict" button.
5. The app will classify the message as either **Spam** or **Ham** and display the result.

Example input:
- "You've won a $1000 gift card! Click here to claim your prize."
- "Hi, are we still meeting at 6 PM?"

The app will classify the first message as **Spam** and the second one as **Ham**.


## How It Works

### Data Preprocessing
- The app uses **NLTK** for text preprocessing. The text is:
  - Converted to lowercase.
  - Tokenized into words.
  - Stop words (common words like "and", "the", etc.) are removed.

### Machine Learning Model
- The model used in this project is a **Random Forest Classifier**. It is trained on a dataset of labeled SMS messages (spam or ham).
- The **TF-IDF Vectorizer** is used to convert the raw text into numerical features that the model can understand.

### Streamlit Interface
- The app uses **Streamlit** for the front-end, which is simple to deploy and highly interactive.
- The user inputs an SMS, and when they click the "Predict" button, the backend processes the text, uses the trained model to predict the result, and then displays the prediction on the screen.


## Technologies Used
- **Streamlit**: A Python library to create web applications with minimal effort.
- **Scikit-learn**: For training and evaluating the machine learning model.
- **NLTK (Natural Language Toolkit)**: For text preprocessing tasks such as tokenization and stopword removal.
- **Random Forest Classifier**: A machine learning algorithm used for classification tasks.
- **TF-IDF Vectorizer**: Used to convert text into a format that the machine learning model can understand.
