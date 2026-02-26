import streamlit as st
import joblib
import re
import string

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

st.title("Sentiment Analysis App")

user_input = st.text_area("Enter review text:")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")
    else:
        st.warning("Please enter text.")
