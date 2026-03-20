import nltk
import streamlit as st
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import os

# Initialize stemmer
stemming = PorterStemmer()

# Download NLTK data once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load model and vectorizer
project_root = os.path.dirname(__file__)
tfidf_path = os.path.join(project_root, 'vectorizer.pkl')
model_path = os.path.join(project_root, 'model.pkl')

with open(tfidf_path, 'rb') as f:
    tfidf = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Text preprocessing
def transform_text(text):
    if not text:  # handle None or empty input
        return ""
    text = str(text).lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    processed = [
        stemming.stem(i) for i in y
        if i not in stopwords.words('english') and i not in string.punctuation
    ]
    return " ".join(processed)

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_msg = st.chat_input("Enter the message")

# Only process if input is not None or empty
if input_msg:
    transformed_txt = transform_text(input_msg)
    vector_input = tfidf.transform([transformed_txt])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
