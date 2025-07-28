import streamlit as st
from modules.data_loader import load_data
from modules.text_cleaner import clean_text
from modules.model_trainer import train_model
from modules.predictor import predict_news
import os
import pickle
import pandas as pd

# Load or train model

#@st.cache_resource
def setup():
    data_fake = load_data("Fake.csv",1)
    data_true = load_data("True.csv",0)
    data = pd.concat([data_fake, data_true], ignore_index=True)
    data['text'] = data['text'].apply(clean_text)
    model, vectorizer, acc = train_model(data['text'], data['label'])
    return model, vectorizer, acc

# Train model
model, vectorizer, acc = setup()
print(f"Model trained with accuracy: {acc * 100:.2f}%")

# UI
st.title("📰 Fake News Detector")
st.markdown(f"**Model Accuracy**: {acc*100:.2f}%")

news_input = st.text_area("Enter News Text", height=200)

if st.button("Check"):
    result = predict_news(news_input, model, vectorizer, clean_text)
    if result == "FAKE":
        st.error("⚠️ This looks like FAKE News.")
    else:
        st.success("✅ This appears to be REAL News.")