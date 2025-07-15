import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
import time
from datetime import datetime
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Mock data and sentiment analysis imports
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Page configuration
st.set_page_config(
    page_title="SentimentFusions - Product Review Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.sentiment_pipeline = None
        self.load_model()
    
    @st.cache_resource
    def load_model(_self):
        """Load the sentiment analysis model"""
        try:
            _self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=_self.model_name,
                tokenizer=_self.model_name,
                return_all_scores=True
            )
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        
        # Remove special characters and emojis
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of a single text"""
        if not self.sentiment_pipeline:
            return "neutral", 0.5
        
        try:
            # Truncate text if too long
            text = text[:512] if len(text) > 512 else text
            
            results = self.sentiment_pipeline(text)[0]
            
            # Map labels to our format
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POSITIVE': 'positive'
            }
            
            best_result = max(results, key=lambda x: x['score'])
            sentiment = label_mapping.get(best_result['label'], 'neutral')
            confidence = best_result['score']
            
            return sentiment, confidence
            
        except Exception as e:
            st.warning(f"Error analyzing text: {str(e)}")
            return "neutral", 0.5

class MockDataGenerator:
    """Generate mock review data for demonstration"""
    
    @staticmethod
    def generate_sample_reviews(product_name, num_reviews=50):
        """Generate sample reviews for a product"""
        
        positive_templates = [
            "Amazing {product}! Highly recommend it. Great quality and fast delivery.",
            "Love this {product}! Exceeded my expectations. Will buy again.",
            "Excellent {product}. Perfect for my needs. 5 stars!",
            "Outstanding quality {product}. Worth every penny. Fast shipping too.",
            "Best {product} I've ever bought. Incredible value for money.",
            "Fantastic {product}! Works perfectly. Great customer service.",
            "Superb {product}. Exactly as described. Very satisfied.",
            "Top-notch {product}. Highly durable and well-made.",
            "Incredible {product}! Solved all my problems. Recommended!",
            "Perfect {product}. Great design and functionality."
        ]
        
        negative_templates = [
            "Terrible {product}. Broke after one week. Waste of money.",
            "Poor quality {product}. Not as advertised. Very disappointed.",
            "Awful {product}. Doesn't work properly. Returning it.",
            "Bad {product}. Cheap materials and poor construction.",
            "Horrible {product}. Completely useless. Don't buy this.",
            "Disappointing {product}. Not worth the price. Poor quality.",
            "Defective {product}. Stopped working immediately. Terrible.",
            "Worst {product} ever. Complete waste of time and money.",
            "Faulty {product}. Multiple issues from day one.",
            "Useless {product}. Nothing like the description. Avoid!"
        ]
        
        neutral_templates = [
            "Okay {product}. Does the job but nothing special.",
            "Average {product}. Some good points, some bad.",
            "Decent {product}. Works as expected. Nothing extraordinary.",
            "Fair {product}. Could be better but acceptable.",
            "Standard {product}. Gets the job done adequately.",
            "Mediocre {product}. Has pros and cons.",
            "Reasonable {product}. Not great, not terrible.",
            "Acceptable {product}. Meets basic requirements.",
            "So-so {product}. Mixed feelings about this purchase.",
            "Regular {product}. Nothing to complain about, nothing to praise."
        ]
        
        reviews = []
        
        # Generate reviews with realistic distribution
        for i in range(num_reviews):
            # 40% positive, 30% negative, 30% neutral
            rand = np.random.random()
            if rand < 0.4:
                template = np.random.choice(positive_templates)
                rating = np.random.choice([4, 5], p=[0.3, 0.7])
            elif rand < 0.7:
                template = np.random.choice(negative_templates)
                rating = np.random.choice([1, 2], p=[0.6, 0.4])
            else:
                template = np.random.choice(neutral_templates)
                rating = 3
            
            review_text = template.format(product=product_name)
            
            # Add some variation
            if np.random.random() < 0.3:
                review_text += f" The delivery was {'fast' if rating >= 4 else 'slow'}."
            
            reviews.append({
                'review_id': f"REV_{i+1:03d}",
                'reviewer_name': f"Customer_{i+1}",
                'rating': rating,
                'review_text': review_text,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'verified_purchase': np.random.choice([True, False], p=[0.8, 0.2])
            })
        
        return pd.DataFrame(reviews)

def create_visualizations(df):
    """Create all visualizations for the sentiment analysis"""
    
    # Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    
    # Pie chart
    fig_pie = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C', 
            'neutral': '#FFD700'
        }
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    
    # Bar chart
    fig_bar = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        title="Sentiment Count Distribution",
        color=sentiment_counts.index,
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#FFD700'
        }
    )
    fig_bar.update_layout(showlegend=False)
    
    # Confidence distribution
    fig_conf = px.histogram(
        df,
        x='confidence',
        color='sentiment',
        title="Confidence Score Distribution by Sentiment",
        nbins=20,
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#FFD700'
        }
    )
    
    return fig_pie, fig_bar, fig_conf

def create_wordclouds(df):
    """Create word clouds for positive and negative reviews"""
    
    positive_text = ' '.join(df[df['sentiment'] == 'positive']['cleaned_text'].astype(str))
    negative_text = ' '.join(df[df['sentiment'] == 'negative']['cleaned_text'].astype(str))
    
    wordclouds = {}
    
    if positive_text.strip():
        wordclouds['positive'] = WordCloud(
            width=400, height=300,
            background_color='white',
            colormap='Greens',
            max_words=50
        ).generate(positive_text)
    
    if negative_text.strip():
        wordclouds['negative'] = WordCloud(
            width=400, height=300,
            background_color='white',
            colormap='Reds',
            max_words=50
        ).generate(negative_text)
    
    return wordclouds

def download_csv(df):
    """Create download link for CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis_results.csv">Download CSV File</a>'
    return href

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 SentimentFusions</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Product Review Sentiment Analyzer</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Theme toggle
        theme = st.selectbox("🎨 Theme", ["Light", "Dark"])
        
        # Product input
        st.subheader("📦 Product Information")
        product_input = st.text_input(
            "Product Name or URL",
            placeholder="Enter product name (e.g., 'iPhone 14', 'Samsung TV')",
            help="Enter a product name to analyze mock reviews"
        )
        
        num_reviews = st.slider("Number of Reviews", 20, 100, 50)
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Reviews", type="primary")
        
        # Model info
        st.subheader("🤖 Model Information")
        st.info("Using: cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    # Main content
    if analyze_button and product_input:
        # Initialize analyzer
        analyzer = SentimentAnalyzer()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Generate mock data
            status_text.text("📊 Generating sample reviews...")
            progress_bar.progress(20)
            
            mock_generator = MockDataGenerator()
            df = mock_generator.generate_sample_reviews(product_input, num_reviews)
            
            # Step 2: Clean text
            status_text.text("🧹 Cleaning review text...")
            progress_bar.progress(40)
            
            df['cleaned_text'] = df['review_text'].apply(analyzer.clean_text)
            
            # Step 3: Analyze sentiment
            status_text.text("🤖 Analyzing sentiment...")
            progress_bar.progress(60)
            
            sentiments = []
            confidences = []
            
            for i, text in enumerate(df['cleaned_text']):
                sentiment, confidence = analyzer.analyze_sentiment(text)
                sentiments.append(sentiment)
                confidences.append(confidence)
                
                # Update progress
                progress = 60 + (i / len(df)) * 30
                progress_bar.progress(int(progress))
            
            df['sentiment'] = sentiments
            df['confidence'] = confidences
            
            # Step 4: Complete
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            time.sleep(1)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Store in session state
            st.session_state.analyzed_data = df
            
            st.success(f"✅ Successfully analyzed {len(df)} reviews for '{product_input}'!")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            return
    
    # Display results if available
    if st.session_state.analyzed_data is not None:
        df = st.session_state.analyzed_data
        
        # Metrics
        st.subheader("📈 Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", len(df))
        with col2:
            positive_pct = (df['sentiment'] == 'positive').mean() * 100
            st.metric("Positive %", f"{positive_pct:.1f}%")
        with col3:
            negative_pct = (df['sentiment'] == 'negative').mean() * 100
            st.metric("Negative %", f"{negative_pct:.1f}%")
        with col4:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Visualizations
        st.subheader("📊 Sentiment Analysis Visualizations")
        
        # Create visualizations
        fig_pie, fig_bar, fig_conf = create_visualizations(df)
        
        # Display charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Word clouds
        st.subheader("☁️ Word Clouds")
        wordclouds = create_wordclouds(df)
        
        if wordclouds:
            col1, col2 = st.columns(2)
            
            if 'positive' in wordclouds:
                with col1:
                    st.subheader("Positive Reviews")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordclouds['positive'], interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            
            if 'negative' in wordclouds:
                with col2:
                    st.subheader("Negative Reviews")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordclouds['negative'], interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
        
        # Reviews table
        st.subheader("📋 Detailed Reviews")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                ["All", "Positive", "Negative", "Neutral"]
            )
        with col2:
            search_term = st.text_input("Search in reviews", placeholder="Enter search term...")
        
        # Apply filters
        filtered_df = df.copy()
        
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter.lower()]
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df['review_text'].str.contains(search_term, case=False, na=False)
            ]
        
        # Display table
        display_columns = ['reviewer_name', 'rating', 'sentiment', 'confidence', 'review_text', 'date']
        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            height=400
        )
        
        # Download button
        st.subheader("💾 Export Data")
        st.markdown(download_csv(df), unsafe_allow_html=True)
        
        # Additional insights
        st.subheader("🔍 Additional Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating vs Sentiment correlation
            rating_sentiment = df.groupby(['rating', 'sentiment']).size().unstack(fill_value=0)
            fig_rating = px.bar(
                rating_sentiment,
                title="Rating vs Sentiment Distribution",
                color_discrete_map={
                    'positive': '#2E8B57',
                    'negative': '#DC143C',
                    'neutral': '#FFD700'
                }
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        
        with col2:
            # Top words by sentiment
            st.write("**Most Common Words by Sentiment**")
            for sentiment in ['positive', 'negative']:
                sentiment_text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'].astype(str))
                words = sentiment_text.split()
                word_freq = pd.Series(words).value_counts().head(5)
                
                st.write(f"*{sentiment.title()}:*")
                for word, count in word_freq.items():
                    if len(word) > 3:  # Filter out short words
                        st.write(f"- {word}: {count}")
    
    else:
        # Welcome message
        st.info("👋 Welcome! Enter a product name in the sidebar and click 'Analyze Reviews' to get started.")
        
        # Feature overview
        st.subheader("🚀 Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Sentiment Analysis**
            - Advanced AI-powered sentiment classification
            - Confidence scores for each prediction
            - Support for positive, negative, and neutral sentiments
            """)
        
        with col2:
            st.markdown("""
            **📈 Visualizations**
            - Interactive pie and bar charts
            - Word clouds for different sentiments
            - Confidence distribution analysis
            """)
        
        with col3:
            st.markdown("""
            **💾 Data Export**
            - Searchable and filterable review table
            - CSV export functionality
            - Detailed review analysis
            """)

if __name__ == "__main__":
    main()