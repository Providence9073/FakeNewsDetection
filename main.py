from modules.data_loader import load_data
from modules.text_cleaner import clean_text
from modules.model_trainer import train_model
from modules.predictor import predict_news
import pandas as pd

# Load & clean data
data_fake = load_data("Fake.csv")
data_true = load_data("True.csv")
data_fake["title"] = 1
data_true["title"] = 0
data = pd.concat([data_fake, data_true], ignore_index=True)
data = data[["text", "title"]].dropna()
data['text'] = data['text'].apply(clean_text)

# Train model
model, vectorizer, acc = train_model(data['text'], data['title'])
print(f"Model trained with accuracy: {acc * 100:.2f}%")

# Test prediction
sample = "COVID-19 vaccine creates magnetic field in body – claims go viral"
result = predict_news(sample, model, vectorizer, clean_text)
print(f"Prediction: {result}")