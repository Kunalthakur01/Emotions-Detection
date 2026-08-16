import streamlit as st
import joblib

import string
import nltk
from nltk.corpus import stopwords

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)

# Load trained model and TF-IDF vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Emotion mapping
emotion_labels = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy"
}


stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = "".join(char for char in text if not char.isdigit())

    # Remove emojis / non-ASCII characters
    text = "".join(char for char in text if char.isascii())

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

def predict_emotion(text):
    cleaned_text = clean_text(text)

    # Convert text into TF-IDF features
    text_tfidf = vectorizer.transform([cleaned_text])

    # Predict emotion
    prediction = model.predict(text_tfidf)[0]

    return emotion_labels[prediction]

st.title("😊 Emotion Detection")
st.caption("AI-powered emotion classification using TF-IDF + Linear SVM")

st.divider()

st.subheader("Enter your text")

text = st.text_area(
    "What are you feeling?",
    placeholder="Example: I am extremely happy today!",
    height=150,
    label_visibility="collapsed"
)

if st.button("🔍 Detect Emotion", use_container_width=True):

    if text.strip():

        emotion = predict_emotion(text)

        st.divider()

        st.subheader("Prediction")

        st.success(f"### {emotion}")

    else:
        st.warning("Please enter some text first.")