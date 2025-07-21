import streamlit as st
import requests
import torch
from plotly import graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

#Retrieving the API key from the .env file
API_KEY = os.getenv("NEWS_API_KEY")
if not API_KEY:
    raise ValueError("API key not found! Make sure it's set in the .env file.")

#feteching the news articles from the News API
def fetch_news(company_name, api_key):
    url = 'https://newsapi.org/v2/everything'
    to_date = datetime.now()
    from_date = to_date - timedelta(days=10)

    params = {
        'q': company_name,
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d'),
        'sortBy': 'relevancy',
        'apiKey': api_key,
        'language': 'en',
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    if data['status'] == 'ok':
        return data['articles']
    else:
        st.error("Error fetching articles")
        return []

# Function to save articles to a DataFrame
def save_to_dataframe(articles):
    data = {
        'date': [article['publishedAt'] for article in articles],
        'content': [article['content'] for article in articles]
    }
    return pd.DataFrame(data)

# Function to analyze sentiment
@st.cache_resource
def load_model_and_tokenizer():
    """Load the FinBERT model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model


def analyze_sentiment(data, tokenizer, model):
    """Perform sentiment analysis using FinBERT."""
    sentiments = []
    scores = []

    for article in data["content"]:
        inputs = tokenizer(article, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs, dim=1).item()

        # Map the sentiment index to labels
        sentiment_labels = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        sentiments.append(sentiment_labels[sentiment])
        scores.append(probs.max().item())

    data["sentiment"] = sentiments
    data["score"] = scores
    return data

# Pie Chart for Sentiment Distribution
def create_sentiment_donuts(df):
    sentiment_counts = df['sentiment'].value_counts()
    total_records = len(df)
    sentiment_percentages = (sentiment_counts / total_records * 100).round(1)
    
    # Colors for each sentiment
    colors = {
        'positive': '#22c55e',  # green
        'negative': '#ef4444',  # red
        'neutral': '#6b7280'    # gray
    }
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    # Function to create individual donut chart
    def create_donut(sentiment, percentage, color):
        fig = go.Figure()
        fig.add_trace(go.Pie(
            values=[percentage, 100-percentage],
            labels=[sentiment, 'other'],
            hole=0.7,
            marker_colors=[color, '#e5e7eb'],
            textinfo='none'
        ))
        
        # Add percentage in center
        fig.update_layout(
            annotations=[{
                'text': f'{percentage}%',
                'x': 0.5,
                'y': 0.5,
                'font_size': 24,
                'showarrow': False
            }],
            showlegend=False,
            width=200,
            height=200,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        return fig
    
    # Create individual donut charts
    with col1:
        st.header("Positive")
        positive_pct = sentiment_percentages.get('POSITIVE')
        fig_positive = create_donut('Positive', positive_pct, colors['positive'])
        st.plotly_chart(fig_positive, use_container_width=True)
        st.metric("Count", sentiment_counts.get('POSITIVE', 0))
    
    with col2:
        st.header("Negative")
        negative_pct = sentiment_percentages.get('NEGATIVE', 0)
        fig_negative = create_donut('Negative', negative_pct, colors['negative'])
        st.plotly_chart(fig_negative, use_container_width=True)
        st.metric("Count", sentiment_counts.get('NEGATIVE', 0))
    
    with col3:
        st.header("Neutral")
        neutral_pct = sentiment_percentages.get('NEUTRAL', 0)
        fig_neutral = create_donut('Neutral', neutral_pct, colors['neutral'])
        st.plotly_chart(fig_neutral, use_container_width=True)
        st.metric("Count", sentiment_counts.get('NEUTRAL', 0))
    
    # Add overall distribution donut chart below
    st.subheader("Overall Distribution")
    fig_overall = px.pie(
        values=sentiment_percentages,
        names=sentiment_percentages.index,
        color=sentiment_percentages.index,
        color_discrete_map=colors,
        hole=0.6
    )
    fig_overall.update_traces(textinfo='percent+label')
    fig_overall.update_layout(
        showlegend=False,
        height=500,
        annotations=[{
            'text': f'Total<br>{total_records}',
            'x': 0.5,
            'y': 0.5,
            'font_size': 20,
            'showarrow': False
        }]
    )
    st.plotly_chart(fig_overall, use_container_width=True)

# Line Chart for Scores over Time
def plot_score_scatter(data):
    """Generate a scatter plot with a trend line for scores over time."""
    data["date"] = pd.to_datetime(data["date"])
    fig = px.scatter(
        data,
        x="date",
        y="score",
        title="Sentiment Scores Over Time with Trend Line",
        labels={"score": "Score", "date": "Date"},
        trendline="lowess",  # Add a smooth trendline
        trendline_color_override="red"
    )
    return fig

def main():
    st.title("Sentiment Analysis of Businesses")

    # Input field for company name
    company_name = st.text_input("Enter the company name:", "")

    if st.button("Analyze News"):
        if not company_name:
            st.warning("Please enter a company name.")
            return

        st.info(f"Fetching news articles about '{company_name}'...")

        # Fetch news articles
        articles = fetch_news(company_name, API_KEY)

        if not articles:
            st.error("No articles found.")
            return

        # Convert articles to DataFrame
        df = save_to_dataframe(articles)

        # Perform sentiment analysis
        tokenizer, model = load_model_and_tokenizer()

        with st.spinner("Analyzing sentiments..."):
            analyzed_data = analyze_sentiment(df, tokenizer, model)
        
        #Donut Charts
        create_sentiment_donuts(df)

        # Display Line Chart for Scores over Time
        st.subheader("Scores Over Time")
        score_chart = plot_score_scatter(analyzed_data)
        st.plotly_chart(score_chart)

if __name__ == "__main__":
    main()