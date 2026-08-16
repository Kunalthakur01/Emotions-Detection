import streamlit as st
import joblib

import string
import nltk
from nltk.corpus import stopwords


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

# -------------------------------
# Frontend UI
# -------------------------------

st.set_page_config(
    page_title="Emotion Detector",
    page_icon="😊",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.result-emotion {
    font-size: 32px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    '<div class="title">😊 Emotion Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect emotions from text using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# Input section
st.subheader("💬 Enter your text")

text = st.text_area(
    "Text",
    placeholder="Example: I am extremely happy today!",
    height=160,
    label_visibility="collapsed"
)


# Prediction
if st.button("🔍 Detect Emotion", use_container_width=True):

    if text.strip():

        emotion = predict_emotion(text)

        st.markdown(
            f"""
            <div class="result-box">
                <div>Detected Emotion</div>
                <div class="result-emotion">{emotion}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("Please enter some text first.")




