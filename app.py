import streamlit as st
import pickle
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def fetch_amazon_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching product details: {e}")
        return "Title Not Found", None, []

    soup = BeautifulSoup(response.content, "html.parser")
    
    title = soup.find("span", {'id': 'productTitle'})
    title = title.text.strip() if title else "Title Not Found"
    
    image = soup.find("img", {'id': 'landingImage'})
    image_url = image['src'] if image else "https://via.placeholder.com/300"
    
    reviews = []
    review_elements = soup.find_all("span", {'data-hook': 'review-body'})
    if not review_elements:
        review_elements = soup.find_all("div", {'class': 'a-expander-content'})
    
    for review in review_elements[:5]:
        reviews.append(review.text.strip())
    
    return title, image_url, reviews

def load_model():
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

def predict_sentiment(model, vectorizer, reviews):
    if not reviews:
        return "Not enough reviews", []
    processed_reviews = vectorizer.transform(reviews)
    if processed_reviews.shape[0] == 0:
        return "Not enough reviews", []
    predictions = model.predict(processed_reviews)
    avg_sentiment = np.mean(predictions) * 10
    
    if avg_sentiment < 2:
        rating = "⭐"
    elif avg_sentiment < 5:
        rating = "⭐⭐"
    elif avg_sentiment < 7:
        rating = "⭐⭐⭐"
    elif avg_sentiment < 9:
        rating = "⭐⭐⭐⭐"
    else:
        rating = "⭐⭐⭐⭐⭐"
    
    return rating, predictions

def generate_wordcloud(reviews):
    text = ' '.join(reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def display_analysis(title, image_url, reviews, predictions):
    st.markdown("## Product Details")
    st.image(image_url, caption="**Product Image**", use_column_width=True)
    
    st.markdown(f"## **{title}**")
    
    st.subheader("Top Reviews:")
    if reviews:
        for review in reviews:
            st.write(f"- {review}")
    else:
        st.write("No reviews found for this product.")
    
    rating, predictions = predict_sentiment(model, vectorizer, reviews)
    
    if isinstance(rating, str):
        st.subheader(f"Predicted Rating: {rating}")
    else:
        st.subheader("Not enough reviews")
    
    st.subheader("Sentiment Distribution")
    sentiment_counts = pd.Series(predictions).value_counts()
    st.bar_chart(sentiment_counts)
    
    st.subheader("Positive Reviews Word Cloud")
    positive_reviews = [reviews[i] for i in range(len(reviews)) if predictions[i] == 1]
    if positive_reviews:
        wordcloud = generate_wordcloud(positive_reviews)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.write("No positive reviews available for word cloud.")
    
    st.subheader("Negative Reviews Word Cloud")
    negative_reviews = [reviews[i] for i in range(len(reviews)) if predictions[i] == 0]
    if negative_reviews:
        wordcloud = generate_wordcloud(negative_reviews)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.write("No negative reviews available for word cloud.")

st.title("Amazon Product Review Analysis")
url = st.text_input("Enter Amazon Product URL:")

if url:
    title, image_url, reviews = fetch_amazon_data(url)
    
    if reviews:
        model, vectorizer = load_model()
        rating, predictions = predict_sentiment(model, vectorizer, reviews)
        
        st.session_state['title'] = title
        st.session_state['image_url'] = image_url
        st.session_state['reviews'] = reviews
        st.session_state['predictions'] = predictions
        
        display_analysis(title, image_url, reviews, predictions)
    else:
        st.write("No reviews found for this product.")
