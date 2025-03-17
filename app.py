import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from datetime import datetime
from textblob import TextBlob
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# Transformers & Semantic Search
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import wikipedia  # For offline events summary
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# --------------------------------------------------------------------------------
# ----------------------- Data Loading and Normalization -------------------------
# --------------------------------------------------------------------------------
@st.cache_data
def load_raw_data(filepath):
    """Load the newline-delimited JSON file into a Pandas DataFrame."""
    try:
        raw_df = pd.read_json(filepath, lines=True)
    except ValueError as e:
        st.error("Error reading the JSONL file. Please check the file format.")
        raise e
    return raw_df

DATA_PATH = "data.jsonl"
if not os.path.exists(DATA_PATH):
    st.error("data.jsonl file not found. Please ensure it is in the same directory as this app.")
else:
    raw_df = load_raw_data(DATA_PATH)

st.sidebar.markdown("### Raw Dataset Columns")
st.sidebar.write(raw_df.columns.tolist())

# Normalize the nested "data" column if present
if 'data' in raw_df.columns:
    try:
        df = pd.json_normalize(raw_df['data'])
    except Exception as e:
        st.error("Error normalizing the 'data' column.")
        df = raw_df
else:
    df = raw_df

st.sidebar.markdown("### Normalized Data Columns")
st.sidebar.write(df.columns.tolist())

# --------------------------------------------------------------------------------
# ------------------------- Column Mapping (Reddit Data) ---------------------------
# --------------------------------------------------------------------------------
# Typical Reddit fields:
timestamp_col = "created_utc"  # Unix timestamp (in seconds)
user_col = "author"            # Author

# For text, prefer "selftext" if available; otherwise, use "title".
if "selftext" in df.columns and df["selftext"].notnull().sum() > 0:
    text_col = "selftext"
elif "title" in df.columns:
    text_col = "title"
else:
    text_col = None

# For hashtags: if not provided, extract from text using regex.
if "hashtags" not in df.columns:
    def extract_hashtags(row):
        text = ""
        if "title" in row and pd.notnull(row["title"]):
            text += row["title"] + " "
        if "selftext" in row and pd.notnull(row["selftext"]):
            text += row["selftext"]
        return re.findall(r"#\w+", text)
    df["hashtags"] = df.apply(extract_hashtags, axis=1)
hashtags_col = "hashtags"

# Convert Unix timestamp to datetime if available
if timestamp_col in df.columns:
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
    except Exception as e:
        st.error(f"Error converting timestamp. Check the format of '{timestamp_col}'.")

# --------------------------------------------------------------------------------
# --------------------------- Sidebar: Filters & Platform --------------------------
# --------------------------------------------------------------------------------
st.sidebar.header("Filters & Platform")

# Platform Selector (simulate multiple platforms)
platform = st.sidebar.selectbox("Select Platform", ["Reddit", "Twitter", "Facebook"])
if platform != "Reddit":
    st.sidebar.info(f"Data for {platform} is not available. Showing Reddit data.")

# Date Filter
if timestamp_col in df.columns:
    try:
        min_date = df[timestamp_col].min().date()
        max_date = df[timestamp_col].max().date()
        start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)
        if start_date > end_date:
            st.sidebar.error("Error: End date must fall after start date.")
        df = df[(df[timestamp_col].dt.date >= start_date) & (df[timestamp_col].dt.date <= end_date)]
    except Exception as e:
        st.sidebar.error("Error processing the timestamp column for filtering.")
else:
    st.sidebar.info(f"No '{timestamp_col}' column found for filtering by date.")

# Keyword/Hashtag Search
search_term = st.sidebar.text_input("Search for a keyword/hashtag:")
if search_term:
    if text_col in df.columns:
        df = df[df[text_col].str.contains(search_term, case=False, na=False)]
    st.sidebar.markdown(f"### Showing results for '{search_term}'")

# --------------------------------------------------------------------------------
# ------------------------- Main Dashboard: Basic Visualizations -----------------
# --------------------------------------------------------------------------------
st.title("Social Media Data Analysis Dashboard")
st.markdown("""
This dashboard visualizes Reddit data, showcasing trends over time, key contributors, topic embeddings, and more.
""")

# Summary Metrics
total_posts = len(df)
st.markdown("### Summary Metrics")
st.write("**Total Posts:**", total_posts)
if user_col in df.columns:
    unique_users = df[user_col].nunique()
    st.write("**Unique Users:**", unique_users)
else:
    st.write("**Unique Users:** Data not available")

# Time Series Plot with 7-day Moving Average
if timestamp_col in df.columns:
    st.markdown("### Posts Over Time with Moving Average")
    df["date"] = df[timestamp_col].dt.date
    time_series = df.groupby("date").size().reset_index(name="count")
    time_series["7-day Moving Avg"] = time_series["count"].rolling(window=7).mean()
    fig_time = px.line(time_series, x="date", y=["count", "7-day Moving Avg"],
                       labels={"date": "Date", "value": "Number of Posts"},
                       title="Posts Over Time with 7-day Moving Average")
    st.plotly_chart(fig_time)
else:
    st.info("No timestamp data available for time series plot.")

# Pie Chart of Top Contributors (using subreddit if available, otherwise author)
community_col = "subreddit" if "subreddit" in df.columns else user_col
if community_col in df.columns:
    st.markdown("### Top Communities/Accounts Contributions")
    contributions = df[community_col].value_counts().reset_index()
    contributions.columns = [community_col, "count"]
    top_contributions = contributions.head(10)
    fig_pie = px.pie(top_contributions, values="count", names=community_col,
                     title="Top 10 Contributors")
    st.plotly_chart(fig_pie)
else:
    st.info("No community or account data available for contributor pie chart.")

# Top Hashtags Bar Chart
if hashtags_col in df.columns:
    st.markdown("### Top Hashtags")
    hashtags_exploded = df.explode(hashtags_col)
    hashtags_exploded = hashtags_exploded[hashtags_exploded[hashtags_col] != ""]
    top_hashtags = hashtags_exploded[hashtags_col].value_counts().reset_index()
    top_hashtags.columns = ['hashtag', 'count']
    if not top_hashtags.empty:
        fig_hashtags = px.bar(top_hashtags.head(10), x='hashtag', y='count',
                              labels={'hashtag': 'Hashtag', 'count': 'Frequency'},
                              title="Top 10 Hashtags")
        st.plotly_chart(fig_hashtags)
    else:
        st.info("No hashtag data available.")
else:
    st.info("No 'hashtags' column found in the dataset.")

# Sentiment Analysis on Text Data
if text_col is not None and text_col in df.columns:
    st.markdown("### Sentiment Analysis")
    df['sentiment'] = df[text_col].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)
    fig_sentiment = px.histogram(df, x='sentiment', nbins=30,
                                 labels={'sentiment': 'Sentiment Polarity'},
                                 title="Sentiment Polarity Distribution")
    st.plotly_chart(fig_sentiment)
else:
    st.info(f"No '{text_col}' column available for sentiment analysis.")

# --------------------------------------------------------------------------------
# ---------------------------- Optional Features ---------------------------------
# Use sidebar checkboxes to toggle optional features
# --------------------------------------------------------------------------------
st.sidebar.markdown("### Optional Features")
show_topic_embedding = st.sidebar.checkbox("Topic Embedding Visualization")
show_ts_genai_summary = st.sidebar.checkbox("GenAI Summary for Time Series")
show_offline_events = st.sidebar.checkbox("Offline Events (Wikipedia)")
show_semantic_search = st.sidebar.checkbox("Semantic Search on Posts")

# ---------------------------------------------------------------------
# (a) Topic Embedding Visualization using LDA + TSNE
# ---------------------------------------------------------------------
if show_topic_embedding:
    st.markdown("## Topic Embedding Visualization")
    if text_col in df.columns:
        texts = df[text_col].dropna().sample(n=min(500, len(df)), random_state=42).tolist()
        vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        topic_matrix = lda.fit_transform(X)
        dominant_topic = topic_matrix.argmax(axis=1)
        tsne_model = TSNE(n_components=2, random_state=42)
        tsne_values = tsne_model.fit_transform(topic_matrix)
        tsne_df = pd.DataFrame(tsne_values, columns=["x", "y"])
        tsne_df["Dominant Topic"] = dominant_topic.astype(str)
        fig_topics = px.scatter(tsne_df, x="x", y="y", color="Dominant Topic",
                                title="TSNE Embedding of Topics")
        st.plotly_chart(fig_topics)
    else:
        st.info("No text data available for topic embedding.")

# ---------------------------------------------------------------------
# (b) GenAI Summary for Time Series Plot
# ---------------------------------------------------------------------
if show_ts_genai_summary:
    st.markdown("## GenAI Summary for Time Series")
    if not time_series.empty:
        start = time_series["date"].min()
        end = time_series["date"].max()
        avg_posts = time_series["count"].mean()
        peak = time_series.loc[time_series["count"].idxmax()]
        description = (f"From {start} to {end}, the average number of posts per day was {avg_posts:.1f}. "
                       f"The highest activity was on {peak['date']} with {peak['count']} posts.")
        st.write("Time Series Description:")
        st.write(description)
        ts_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        try:
            ts_summary = ts_summarizer(description, max_length=80, min_length=40, do_sample=False)[0]['summary_text']
            st.markdown("**GenAI Summary:**")
            st.write(ts_summary)
        except Exception as e:
            st.error("Error generating time series summary.")
    else:
        st.info("Time series data not available for summarization.")

# ---------------------------------------------------------------------
# (d) Offline Events from Wikipedia for a Given Topic
# ---------------------------------------------------------------------
if show_offline_events:
    st.markdown("## Offline Events from Wikipedia")
    wiki_topic = st.text_input("Enter a topic to fetch offline events (e.g., 'Russian invasion of Ukraine'):")
    if wiki_topic:
        try:
            wiki_summary = wikipedia.summary(wiki_topic, sentences=5)
            st.markdown(f"**Wikipedia Summary for '{wiki_topic}':**")
            st.write(wiki_summary)
        except Exception as e:
            st.error("Error retrieving Wikipedia data. Please check the topic name.")

# ---------------------------------------------------------------------
# (f) Semantic Search on Posts using Sentence Transformers
# ---------------------------------------------------------------------
if show_semantic_search:
    st.markdown("## Semantic Search on Posts")
    search_query = st.text_input("Enter your semantic search query:")
    if search_query and text_col in df.columns:
        @st.cache_data
        def get_post_embeddings(texts):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model.encode(texts, convert_to_tensor=True)
        posts = df[text_col].dropna().tolist()
        embeddings = get_post_embeddings(posts)
        query_embedding = SentenceTransformer("all-MiniLM-L6-v2").encode(search_query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        top_results = cos_scores.topk(5)
        st.markdown("**Top Matching Posts:**")
        for score, idx in zip(top_results.values, top_results.indices):
            st.write(f"Score: {score.item():.3f}")
            st.write(posts[idx])
            st.write("---")

# ---------------------------------------------------------------------
# (Optional) AI-Generated Summary on Posts (Existing Feature)
# ---------------------------------------------------------------------
st.markdown("## AI-Generated Summary of Posts")
if text_col in df.columns:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    def generate_summary(text, summarizer, max_chunk_length=1000):
        chunks, current_chunk = [], ""
        for sentence in text.split('. '):
            sentence = sentence.strip() + ". "
            if len(current_chunk) + len(sentence) <= max_chunk_length:
                current_chunk += sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        summaries = []
        for chunk in chunks:
            if len(chunk) > 50:
                summary_chunk = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
                summaries.append(summary_chunk)
        combined_summary = " ".join(summaries)
        final_summary = summarizer(combined_summary, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        return final_summary

    sample_text = " ".join(df[text_col].dropna().sample(n=min(10, len(df)), random_state=42).tolist())
    if sample_text:
        final_summary = generate_summary(sample_text, summarizer, max_chunk_length=1000)
        st.write(final_summary)
    else:
        st.info("Not enough text data available for summarization.")
else:
    st.info("No text data available for AI summarization.")

# --------------------------------------------------------------------------------
# ------------------------------- End of Dashboard -------------------------------
# --------------------------------------------------------------------------------
st.markdown("### End of Dashboard")
st.markdown("""
This dashboard is a prototype implementation for analyzing Reddit social media data.  
It demonstrates advanced trend analysis, contributor insights, topic embeddings, GenAI summaries, offline event linking, and semantic search functionality.
""")
