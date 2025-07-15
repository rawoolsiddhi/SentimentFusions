import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
import time
from datetime import datetime, timedelta
import base64
from io import BytesIO
import warnings
import random
warnings.filterwarnings('ignore')

# Mock data and sentiment analysis imports
from transformers import pipeline
import torch

# Page configuration
st.set_page_config(
    page_title="SentimentFusions - Product Review Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and theme support
def load_css():
    st.markdown("""
    <style>
        /* Main styling */
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
        
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: bold;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            transition: border-color 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Table styling */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Success/Error message styling */
        .stSuccess {
            border-radius: 10px;
            border-left: 5px solid #28a745;
        }
        
        .stError {
            border-radius: 10px;
            border-left: 5px solid #dc3545;
        }
        
        /* Dark theme adjustments */
        @media (prefers-color-scheme: dark) {
            .subtitle {
                color: #ccc;
            }
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            
            .metric-card {
                padding: 1rem;
            }
        }
        
        /* Animation for loading */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 2s infinite;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

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
            "Perfect {product}. Great design and functionality.",
            "Wonderful {product}. Easy to use and very effective.",
            "Brilliant {product}! Couldn't be happier with this purchase."
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
            "Useless {product}. Nothing like the description. Avoid!",
            "Broken {product}. Arrived damaged and doesn't work.",
            "Overpriced {product}. Poor quality for the money."
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
        reviewer_names = [
            "John D.", "Sarah M.", "Mike R.", "Lisa K.", "David W.",
            "Emma S.", "Chris P.", "Anna L.", "Tom B.", "Maria G.",
            "Alex J.", "Sophie T.", "Ryan H.", "Kate F.", "Mark C."
        ]
        
        # Generate reviews with realistic distribution
        for i in range(num_reviews):
            # 45% positive, 25% negative, 30% neutral
            rand = np.random.random()
            if rand < 0.45:
                template = np.random.choice(positive_templates)
                rating = np.random.choice([4, 5], p=[0.3, 0.7])
            elif rand < 0.70:
                template = np.random.choice(negative_templates)
                rating = np.random.choice([1, 2], p=[0.6, 0.4])
            else:
                template = np.random.choice(neutral_templates)
                rating = 3
            
            review_text = template.format(product=product_name)
            
            # Add some variation
            if np.random.random() < 0.3:
                variations = [
                    f" The delivery was {'fast' if rating >= 4 else 'slow'}.",
                    f" Customer service was {'helpful' if rating >= 4 else 'poor'}.",
                    f" Would {'definitely' if rating >= 4 else 'not'} recommend."
                ]
                review_text += np.random.choice(variations)
            
            # Generate random date within last 6 months
            days_ago = random.randint(1, 180)
            review_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            reviews.append({
                'review_id': f"REV_{i+1:03d}",
                'reviewer_name': random.choice(reviewer_names),
                'rating': rating,
                'review_text': review_text,
                'date': review_date,
                'verified_purchase': np.random.choice([True, False], p=[0.8, 0.2])
            })
        
        return pd.DataFrame(reviews)

def create_sentiment_visualizations(df):
    """Create sentiment analysis visualizations"""
    
    # Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    
    # Color scheme
    colors = {
        'positive': '#2E8B57',
        'negative': '#DC143C', 
        'neutral': '#FFD700'
    }
    
    # Pie chart
    fig_pie = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="<b>Sentiment Distribution</b>",
        color=sentiment_counts.index,
        color_discrete_map=colors,
        hole=0.4
    )
    fig_pie.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont_size=12,
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    fig_pie.update_layout(
        font=dict(size=14),
        showlegend=True,
        height=400,
        title_x=0.5
    )
    
    # Bar chart
    fig_bar = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        title="<b>Sentiment Count Distribution</b>",
        color=sentiment_counts.index,
        color_discrete_map=colors,
        text=sentiment_counts.values
    )
    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
    fig_bar.update_layout(
        showlegend=False,
        xaxis_title="Sentiment",
        yaxis_title="Number of Reviews",
        font=dict(size=14),
        height=400,
        title_x=0.5
    )
    
    return fig_pie, fig_bar

def create_confidence_visualization(df):
    """Create confidence score distribution visualization"""
    fig_conf = px.histogram(
        df,
        x='confidence',
        color='sentiment',
        title="<b>Confidence Score Distribution by Sentiment</b>",
        nbins=20,
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#FFD700'
        },
        opacity=0.7
    )
    fig_conf.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        font=dict(size=14),
        height=400,
        title_x=0.5,
        bargap=0.1
    )
    return fig_conf

def create_rating_sentiment_correlation(df):
    """Create rating vs sentiment correlation visualization"""
    if 'rating' in df.columns:
        rating_sentiment = df.groupby(['rating', 'sentiment']).size().unstack(fill_value=0)
        
        fig_rating = px.bar(
            rating_sentiment,
            title="<b>Rating vs Sentiment Distribution</b>",
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C',
                'neutral': '#FFD700'
            },
            barmode='stack'
        )
        fig_rating.update_layout(
            xaxis_title="Rating",
            yaxis_title="Number of Reviews",
            font=dict(size=14),
            height=400,
            title_x=0.5
        )
        return fig_rating
    return None

def create_wordclouds(df):
    """Create word clouds for positive and negative reviews"""
    
    positive_text = ' '.join(df[df['sentiment'] == 'positive']['cleaned_text'].astype(str))
    negative_text = ' '.join(df[df['sentiment'] == 'negative']['cleaned_text'].astype(str))
    
    wordclouds = {}
    
    if positive_text.strip():
        wordclouds['positive'] = WordCloud(
            width=600, height=400,
            background_color='white',
            colormap='Greens',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(positive_text)
    
    if negative_text.strip():
        wordclouds['negative'] = WordCloud(
            width=600, height=400,
            background_color='white',
            colormap='Reds',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(negative_text)
    
    return wordclouds

def download_csv(df):
    """Create download link for CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis_results.csv" style="text-decoration: none;"><button style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">📥 Download CSV File</button></a>'
    return href

def display_metrics(df):
    """Display key metrics in cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_reviews = len(df)
    positive_pct = (df['sentiment'] == 'positive').mean() * 100
    negative_pct = (df['sentiment'] == 'negative').mean() * 100
    avg_confidence = df['confidence'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Reviews</h3>
            <h2>{total_reviews}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>😊 Positive</h3>
            <h2>{positive_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>😞 Negative</h3>
            <h2>{negative_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Avg Confidence</h3>
            <h2>{avg_confidence:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Load CSS
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">📊 SentimentFusions</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Product Review Sentiment Analyzer</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Theme toggle
        theme_options = ["Light 🌞", "Dark 🌙"]
        selected_theme = st.selectbox("🎨 Theme", theme_options)
        
        st.markdown("---")
        
        # Product input section
        st.subheader("📦 Product Analysis")
        
        # Product input with better styling
        product_input = st.text_input(
            "🔍 Product Name or URL",
            placeholder="e.g., iPhone 14, Samsung Galaxy, MacBook Pro",
            help="Enter a product name to analyze mock reviews or paste a product URL"
        )
        
        # Number of reviews slider
        num_reviews = st.slider(
            "📊 Number of Reviews to Analyze", 
            min_value=20, 
            max_value=100, 
            value=50,
            help="Select how many reviews to generate and analyze"
        )
        
        st.markdown("---")
        
        # Analysis button with better styling
        analyze_button = st.button(
            "🔍 Analyze Reviews", 
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Model information
        st.subheader("🤖 AI Model Info")
        st.info("""
        **Model**: cardiffnlp/twitter-roberta-base-sentiment-latest
        
        **Capabilities**:
        - Sentiment Classification
        - Confidence Scoring
        - Multi-language Support
        """)
        
        # Tips section
        st.subheader("💡 Tips")
        st.markdown("""
        - Try different product names
        - Adjust review count for faster/detailed analysis
        - Use search and filters in results table
        - Download results as CSV for further analysis
        """)
    
    # Main content area
    if analyze_button and product_input:
        # Initialize analyzer
        with st.spinner("🤖 Loading AI model..."):
            analyzer = SentimentAnalyzer()
        
        if analyzer.sentiment_pipeline is None:
            st.error("❌ Failed to load sentiment analysis model. Please try again.")
            return
        
        # Create progress tracking
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Analysis Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Step 1: Generate mock data
            status_text.markdown("📊 **Generating sample reviews...**")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            mock_generator = MockDataGenerator()
            df = mock_generator.generate_sample_reviews(product_input, num_reviews)
            
            # Step 2: Clean text
            status_text.markdown("🧹 **Cleaning review text...**")
            progress_bar.progress(40)
            time.sleep(0.5)
            
            df['cleaned_text'] = df['review_text'].apply(analyzer.clean_text)
            
            # Step 3: Analyze sentiment
            status_text.markdown("🤖 **Analyzing sentiment with AI...**")
            progress_bar.progress(60)
            
            sentiments = []
            confidences = []
            
            # Process reviews with progress updates
            for i, text in enumerate(df['cleaned_text']):
                sentiment, confidence = analyzer.analyze_sentiment(text)
                sentiments.append(sentiment)
                confidences.append(confidence)
                
                # Update progress
                progress = 60 + (i / len(df)) * 30
                progress_bar.progress(int(progress))
                
                # Small delay for visual effect
                if i % 10 == 0:
                    time.sleep(0.1)
            
            df['sentiment'] = sentiments
            df['confidence'] = confidences
            
            # Step 4: Complete
            status_text.markdown("✅ **Analysis complete!**")
            progress_bar.progress(100)
            time.sleep(1)
            
            # Clear progress indicators
            progress_container.empty()
            
            # Store in session state
            st.session_state.analyzed_data = df
            st.session_state.analysis_complete = True
            
            # Success message
            st.success(f"✅ Successfully analyzed {len(df)} reviews for **{product_input}**!")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            return
    
    # Display results if available
    if st.session_state.analyzed_data is not None and st.session_state.analysis_complete:
        df = st.session_state.analyzed_data
        
        st.markdown("---")
        
        # Display metrics
        st.markdown("## 📈 Analysis Summary")
        display_metrics(df)
        
        st.markdown("---")
        
        # Visualizations section
        st.markdown("## 📊 Sentiment Analysis Visualizations")
        
        # Create and display sentiment charts
        fig_pie, fig_bar = create_sentiment_visualizations(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Confidence distribution
        fig_conf = create_confidence_visualization(df)
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Rating vs sentiment correlation (bonus feature)
        if 'rating' in df.columns:
            fig_rating = create_rating_sentiment_correlation(df)
            if fig_rating:
                st.plotly_chart(fig_rating, use_container_width=True)
        
        st.markdown("---")
        
        # Word clouds section
        st.markdown("## ☁️ Word Cloud Analysis")
        
        with st.spinner("Generating word clouds..."):
            wordclouds = create_wordclouds(df)
        
        if wordclouds:
            col1, col2 = st.columns(2)
            
            if 'positive' in wordclouds:
                with col1:
                    st.markdown("### 😊 Positive Reviews Word Cloud")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordclouds['positive'], interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
            
            if 'negative' in wordclouds:
                with col2:
                    st.markdown("### 😞 Negative Reviews Word Cloud")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordclouds['negative'], interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
        else:
            st.info("No word clouds generated. Need more diverse review data.")
        
        st.markdown("---")
        
        # Reviews table section
        st.markdown("## 📋 Detailed Review Analysis")
        
        # Filter and search options
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            sentiment_filter = st.selectbox(
                "🔍 Filter by Sentiment",
                ["All", "Positive", "Negative", "Neutral"],
                help="Filter reviews by sentiment classification"
            )
        
        with col2:
            search_term = st.text_input(
                "🔎 Search in Reviews", 
                placeholder="Enter keywords to search...",
                help="Search for specific words or phrases in reviews"
            )
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            show_confidence = st.checkbox("Show Confidence", value=True)
        
        # Apply filters
        filtered_df = df.copy()
        
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter.lower()]
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df['review_text'].str.contains(search_term, case=False, na=False)
            ]
        
        # Prepare display columns
        display_columns = ['reviewer_name', 'rating', 'sentiment', 'review_text', 'date']
        if show_confidence:
            display_columns.insert(3, 'confidence')
        
        # Display filtered results count
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} reviews**")
        
        # Display table with custom styling
        if len(filtered_df) > 0:
            # Format confidence scores
            if 'confidence' in filtered_df.columns:
                filtered_df['confidence'] = filtered_df['confidence'].round(3)
            
            st.dataframe(
                filtered_df[display_columns],
                use_container_width=True,
                height=500,
                column_config={
                    "sentiment": st.column_config.TextColumn(
                        "Sentiment",
                        help="AI-predicted sentiment"
                    ),
                    "confidence": st.column_config.NumberColumn(
                        "Confidence",
                        help="Model confidence score (0-1)",
                        format="%.3f"
                    ),
                    "rating": st.column_config.NumberColumn(
                        "Rating",
                        help="User rating (1-5 stars)"
                    ),
                    "review_text": st.column_config.TextColumn(
                        "Review Text",
                        help="Original review content"
                    )
                }
            )
        else:
            st.warning("No reviews match your current filters. Try adjusting your search criteria.")
        
        st.markdown("---")
        
        # Download section
        st.markdown("## 💾 Export Analysis Results")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("Download the complete analysis results including all reviews, sentiment predictions, and confidence scores.")
        
        with col2:
            st.markdown(download_csv(df), unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("---")
        st.markdown("## 🔍 Additional Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Sentiment Summary")
            sentiment_summary = df['sentiment'].value_counts()
            for sentiment, count in sentiment_summary.items():
                percentage = (count / len(df)) * 100
                st.markdown(f"**{sentiment.title()}**: {count} reviews ({percentage:.1f}%)")
        
        with col2:
            st.markdown("### 🎯 Confidence Analysis")
            high_conf = (df['confidence'] >= 0.8).sum()
            medium_conf = ((df['confidence'] >= 0.6) & (df['confidence'] < 0.8)).sum()
            low_conf = (df['confidence'] < 0.6).sum()
            
            st.markdown(f"**High Confidence (≥0.8)**: {high_conf} reviews")
            st.markdown(f"**Medium Confidence (0.6-0.8)**: {medium_conf} reviews")
            st.markdown(f"**Low Confidence (<0.6)**: {low_conf} reviews")
    
    else:
        # Welcome screen
        st.markdown("## 👋 Welcome to SentimentFusions!")
        
        st.markdown("""
        ### 🚀 Get Started
        1. **Enter a product name** in the sidebar (e.g., "iPhone 14", "Samsung TV")
        2. **Adjust the number of reviews** to analyze (20-100)
        3. **Click "Analyze Reviews"** to start the AI-powered analysis
        4. **Explore the results** with interactive charts and detailed insights
        """)
        
        # Feature showcase
        st.markdown("---")
        st.markdown("## ✨ Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 AI-Powered Analysis
            - Advanced sentiment classification
            - Confidence scoring for each prediction
            - Support for positive, negative, and neutral sentiments
            - State-of-the-art transformer models
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Rich Visualizations
            - Interactive pie and bar charts
            - Word clouds for sentiment analysis
            - Confidence distribution analysis
            - Rating vs sentiment correlation
            """)
        
        with col3:
            st.markdown("""
            ### 🔧 Advanced Features
            - Searchable and filterable review table
            - CSV export functionality
            - Responsive design for all devices
            - Real-time progress tracking
            """)
        
        # Demo section
        st.markdown("---")
        st.markdown("## 🎯 Try These Examples")
        
        example_products = [
            "iPhone 14 Pro", "Samsung Galaxy S23", "MacBook Pro", 
            "Sony WH-1000XM4", "Tesla Model 3", "Nintendo Switch"
        ]
        
        cols = st.columns(3)
        for i, product in enumerate(example_products):
            with cols[i % 3]:
                if st.button(f"📱 {product}", key=f"example_{i}"):
                    st.session_state.example_product = product
                    st.rerun()

if __name__ == "__main__":
    main()