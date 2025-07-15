"""
Configuration settings for the sentiment analyzer
"""

# Model configuration
MODEL_CONFIG = {
    'default_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'alternative_models': [
        'nlptown/bert-base-multilingual-uncased-sentiment',
        'cardiffnlp/twitter-roberta-base-sentiment',
        'distilbert-base-uncased-finetuned-sst-2-english'
    ],
    'max_length': 512,
    'truncation': True
}

# UI configuration
UI_CONFIG = {
    'page_title': 'SentimentFusions - Product Review Analyzer',
    'page_icon': '📊',
    'layout': 'wide',
    'theme_colors': {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'positive': '#2E8B57',
        'negative': '#DC143C',
        'neutral': '#FFD700'
    }
}

# Data configuration
DATA_CONFIG = {
    'min_reviews': 20,
    'max_reviews': 100,
    'default_reviews': 50,
    'sentiment_distribution': {
        'positive': 0.45,
        'negative': 0.25,
        'neutral': 0.30
    }
}

# Export configuration
EXPORT_CONFIG = {
    'csv_filename': 'sentiment_analysis_results.csv',
    'include_columns': [
        'review_id', 'reviewer_name', 'rating', 'review_text',
        'cleaned_text', 'sentiment', 'confidence', 'date'
    ]
}

# Performance configuration
PERFORMANCE_CONFIG = {
    'batch_size': 10,
    'cache_ttl': 3600,  # 1 hour
    'max_text_length': 1000
}