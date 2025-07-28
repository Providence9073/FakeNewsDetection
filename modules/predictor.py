def predict_news(news_text, model, vectorizer, cleaner):
    clean = cleaner(news_text)
    vectorized = vectorizer.transform([clean])
    prediction = model.predict(vectorized)
    return prediction[0]