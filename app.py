import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from nltk.corpus import stopwords

# Load the pre-trained model and vectorizer
with open('spam_detector_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

st.title("Spam Detection App")
st.write("This app classifies messages as either Spam or Ham (Not Spam).")

# Input box for user to enter a message
user_message = st.text_area("Enter your message:")

if st.button("Classify"):
    # Clean and preprocess the user input
    clean_message = clean_text(user_message)
    message_tfidf = vectorizer.transform([clean_message])

    # Predict the category
    prediction = model.predict(message_tfidf)
    category = 'Spam' if prediction[0] == 1 else 'Ham'

    # Display the result
    st.write(f'The message is classified as: **{category}**')
