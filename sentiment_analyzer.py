"""
Core sentiment analysis functionality
"""
import pandas as pd
import numpy as np
import re
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """Main sentiment analysis class"""
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the sentiment analysis model"""
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True
            )
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
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
            print(f"Error analyzing text: {str(e)}")
            return "neutral", 0.5
    
    def analyze_batch(self, texts):
        """Analyze sentiment for a batch of texts"""
        results = []
        for text in texts:
            cleaned_text = self.clean_text(text)
            sentiment, confidence = self.analyze_sentiment(cleaned_text)
            results.append({
                'original_text': text,
                'cleaned_text': cleaned_text,
                'sentiment': sentiment,
                'confidence': confidence
            })
        return results

def calculate_sentiment_metrics(df):
    """Calculate various sentiment metrics"""
    metrics = {}
    
    # Basic counts
    sentiment_counts = df['sentiment'].value_counts()
    total_reviews = len(df)
    
    metrics['total_reviews'] = total_reviews
    metrics['positive_count'] = sentiment_counts.get('positive', 0)
    metrics['negative_count'] = sentiment_counts.get('negative', 0)
    metrics['neutral_count'] = sentiment_counts.get('neutral', 0)
    
    # Percentages
    metrics['positive_percentage'] = (metrics['positive_count'] / total_reviews) * 100
    metrics['negative_percentage'] = (metrics['negative_count'] / total_reviews) * 100
    metrics['neutral_percentage'] = (metrics['neutral_count'] / total_reviews) * 100
    
    # Confidence metrics
    metrics['avg_confidence'] = df['confidence'].mean()
    metrics['confidence_std'] = df['confidence'].std()
    
    # Sentiment score (positive - negative)
    metrics['sentiment_score'] = metrics['positive_percentage'] - metrics['negative_percentage']
    
    return metrics