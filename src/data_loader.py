"""
Data Loading and Management for Real Datasets
Supports Kaggle datasets and custom CSV files
"""
import pandas as pd
import numpy as np
import os
import requests
import zipfile
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import logging
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Advanced data loader for sentiment analysis datasets"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.logger = self._setup_logger()
        os.makedirs(data_dir, exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            'kaggle_product_reviews': {
                'url': 'https://www.kaggle.com/datasets/mansithummar67/171k-product-review-with-sentiment-dataset',
                'file_name': 'product_reviews_171k.csv',
                'text_column': 'review_text',
                'sentiment_column': 'sentiment',
                'expected_sentiments': ['positive', 'negative', 'neutral']
            },
            'imdb': {
                'url': 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                'file_name': 'imdb_reviews.csv',
                'text_column': 'review',
                'sentiment_column': 'sentiment',
                'expected_sentiments': ['positive', 'negative']
            },
            'sentiment140': {
                'url': 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip',
                'file_name': 'sentiment140.csv',
                'text_column': 'text',
                'sentiment_column': 'sentiment',
                'expected_sentiments': ['positive', 'negative', 'neutral']
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for data operations"""
        logger = logging.getLogger('DataLoader')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_kaggle_product_reviews(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the Kaggle 171K Product Reviews dataset
        
        Args:
            file_path: Path to the CSV file. If None, looks for default file
            
        Returns:
            DataFrame with cleaned and processed reviews
        """
        self.logger.info("Loading Kaggle Product Reviews dataset...")
        
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'product_reviews_171k.csv')
        
        if not os.path.exists(file_path):
            self.logger.warning(f"Dataset file not found at {file_path}")
            self.logger.info("Please download the dataset from:")
            self.logger.info("https://www.kaggle.com/datasets/mansithummar67/171k-product-review-with-sentiment-dataset")
            self.logger.info("And place it in the 'data' directory")
            
            # Generate synthetic data as fallback
            return self._generate_synthetic_kaggle_like_data()
        
        try:
            # Load the dataset
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} reviews from Kaggle dataset")
            
            # Inspect the dataset structure
            self.logger.info(f"Dataset columns: {list(df.columns)}")
            self.logger.info(f"Dataset shape: {df.shape}")
            
            # Clean and standardize the dataset
            df = self._clean_kaggle_dataset(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading Kaggle dataset: {str(e)}")
            self.logger.info("Generating synthetic data as fallback...")
            return self._generate_synthetic_kaggle_like_data()
    
    def _clean_kaggle_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the Kaggle dataset"""
        
        # Try to identify the correct columns
        text_columns = ['review_text', 'review', 'text', 'Review', 'Text']
        sentiment_columns = ['sentiment', 'Sentiment', 'label', 'Label']
        
        text_col = None
        sentiment_col = None
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        for col in sentiment_columns:
            if col in df.columns:
                sentiment_col = col
                break
        
        if text_col is None or sentiment_col is None:
            self.logger.warning("Could not identify text and sentiment columns")
            self.logger.info(f"Available columns: {list(df.columns)}")
            # Use first two columns as fallback
            text_col = df.columns[0]
            sentiment_col = df.columns[1]
        
        # Standardize column names
        df = df.rename(columns={text_col: 'text', sentiment_col: 'sentiment'})
        
        # Keep only necessary columns
        df = df[['text', 'sentiment']].copy()
        
        # Remove missing values
        df = df.dropna(subset=['text', 'sentiment'])
        
        # Clean text data
        df['text'] = df['text'].astype(str)
        df['text'] = df['text'].str.strip()
        
        # Remove empty texts
        df = df[df['text'].str.len() > 10]  # At least 10 characters
        
        # Standardize sentiment labels
        df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
        
        # Map various sentiment formats to standard format
        sentiment_mapping = {
            'positive': 'positive',
            'pos': 'positive',
            '1': 'positive',
            '4': 'positive',  # Sentiment140 format
            '5': 'positive',
            'negative': 'negative',
            'neg': 'negative',
            '0': 'negative',
            '2': 'negative',
            'neutral': 'neutral',
            '3': 'neutral',
            'mixed': 'neutral'
        }
        
        df['sentiment'] = df['sentiment'].map(sentiment_mapping)
        
        # Remove unmapped sentiments
        df = df[df['sentiment'].notna()]
        
        # Balance the dataset to prevent class imbalance
        df = self._balance_dataset(df)
        
        self.logger.info(f"Cleaned dataset: {len(df)} reviews")
        self.logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        
        return df
    
    def _balance_dataset(self, df: pd.DataFrame, max_samples_per_class: int = 50000) -> pd.DataFrame:
        """Balance the dataset to prevent class imbalance issues"""
        
        sentiment_counts = df['sentiment'].value_counts()
        self.logger.info(f"Original distribution: {dict(sentiment_counts)}")
        
        # Find the minimum class size (but cap it at max_samples_per_class)
        min_class_size = min(sentiment_counts.min(), max_samples_per_class)
        
        balanced_dfs = []
        for sentiment in df['sentiment'].unique():
            sentiment_df = df[df['sentiment'] == sentiment]
            
            if len(sentiment_df) > min_class_size:
                # Downsample
                sentiment_df = resample(sentiment_df, 
                                      n_samples=min_class_size, 
                                      random_state=42)
            elif len(sentiment_df) < min_class_size * 0.5:
                # Upsample if significantly underrepresented
                sentiment_df = resample(sentiment_df, 
                                      n_samples=int(min_class_size * 0.8), 
                                      random_state=42)
            
            balanced_dfs.append(sentiment_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.logger.info(f"Balanced distribution: {dict(balanced_df['sentiment'].value_counts())}")
        
        return balanced_df
    
    def _generate_synthetic_kaggle_like_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic data that mimics the Kaggle dataset structure"""
        
        self.logger.info(f"Generating {n_samples} synthetic product reviews...")
        
        # Product categories
        products = [
            "smartphone", "laptop", "headphones", "tablet", "smartwatch",
            "camera", "speaker", "keyboard", "mouse", "monitor",
            "television", "gaming console", "fitness tracker", "earbuds",
            "power bank", "wireless charger", "smart home device"
        ]
        
        # Positive review templates
        positive_templates = [
            "This {product} is absolutely amazing! The quality is outstanding and it works perfectly. Highly recommend it to everyone.",
            "Excellent {product}! Great value for money and fantastic performance. Very satisfied with this purchase.",
            "Love this {product}! Easy to use, great design, and excellent build quality. Will definitely buy again.",
            "Outstanding {product} with incredible features. Fast delivery and perfect packaging. 5 stars!",
            "Best {product} I've ever owned! Exceeds all expectations and works flawlessly. Highly recommended!",
            "Fantastic {product} with amazing performance. Great customer service and fast shipping. Very happy!",
            "Perfect {product} for my needs. High quality materials and excellent functionality. Love it!",
            "Incredible {product} that works better than expected. Great design and very user-friendly.",
            "Amazing {product} with excellent build quality. Fast performance and great value. Recommended!",
            "Superb {product} that delivers on all promises. Outstanding quality and perfect functionality."
        ]
        
        # Negative review templates
        negative_templates = [
            "Terrible {product}! Poor quality and doesn't work as advertised. Complete waste of money.",
            "Awful {product} that broke after just one week. Poor build quality and terrible customer service.",
            "Horrible {product} with many defects. Doesn't work properly and very disappointing quality.",
            "Worst {product} ever! Completely useless and overpriced. Would not recommend to anyone.",
            "Poor quality {product} that failed immediately. Bad design and terrible performance. Avoid!",
            "Disappointing {product} with multiple issues. Poor materials and doesn't meet expectations.",
            "Defective {product} that arrived damaged. Poor packaging and terrible build quality.",
            "Useless {product} that doesn't work at all. Poor design and complete waste of time.",
            "Terrible {product} with poor performance. Overpriced and doesn't deliver on promises.",
            "Awful {product} that broke immediately. Poor quality control and bad customer support."
        ]
        
        # Neutral review templates
        neutral_templates = [
            "Average {product} that does the job. Nothing special but works as expected for the price.",
            "Okay {product} with mixed results. Some good features but also some limitations.",
            "Decent {product} for basic use. Could be better but acceptable for the price range.",
            "Standard {product} with typical performance. Neither impressive nor disappointing.",
            "Fair {product} that meets basic requirements. Some pros and cons but overall acceptable.",
            "Mediocre {product} with average quality. Works fine but nothing extraordinary.",
            "Regular {product} that does what it's supposed to do. Average build quality and performance.",
            "Acceptable {product} for casual use. Could use some improvements but it's okay.",
            "So-so {product} with mixed feelings. Some aspects are good, others could be better.",
            "Moderate {product} that's neither great nor terrible. Average performance overall."
        ]
        
        # Generate synthetic data
        data = []
        n_per_sentiment = n_samples // 3
        
        # Generate positive reviews
        for _ in range(n_per_sentiment):
            product = np.random.choice(products)
            template = np.random.choice(positive_templates)
            text = template.format(product=product)
            data.append({'text': text, 'sentiment': 'positive'})
        
        # Generate negative reviews
        for _ in range(n_per_sentiment):
            product = np.random.choice(products)
            template = np.random.choice(negative_templates)
            text = template.format(product=product)
            data.append({'text': text, 'sentiment': 'negative'})
        
        # Generate neutral reviews
        for _ in range(n_samples - 2 * n_per_sentiment):
            product = np.random.choice(products)
            template = np.random.choice(neutral_templates)
            text = template.format(product=product)
            data.append({'text': text, 'sentiment': 'neutral'})
        
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.logger.info(f"Generated synthetic dataset with {len(df)} samples")
        
        return df
    
    def create_train_val_test_split(self, df: pd.DataFrame, 
                                   train_size: float = 0.7,
                                   val_size: float = 0.15,
                                   test_size: float = 0.15,
                                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/validation/test splits
        
        Args:
            df: Input dataframe
            train_size: Proportion for training set
            val_size: Proportion for validation set  
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_size + test_size),
            random_state=random_state,
            stratify=df['sentiment']
        )
        
        # Second split: val vs test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / (val_size + test_size),
            random_state=random_state,
            stratify=temp_df['sentiment']
        )
        
        self.logger.info(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        
        stats = {
            'total_samples': len(df),
            'sentiment_distribution': dict(df['sentiment'].value_counts()),
            'avg_text_length': df['text'].str.len().mean(),
            'median_text_length': df['text'].str.len().median(),
            'min_text_length': df['text'].str.len().min(),
            'max_text_length': df['text'].str.len().max(),
            'unique_texts': df['text'].nunique(),
            'duplicate_rate': (len(df) - df['text'].nunique()) / len(df) * 100
        }
        
        return stats
    
    def save_processed_dataset(self, df: pd.DataFrame, filename: str) -> str:
        """Save processed dataset to file"""
        
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved processed dataset to {filepath}")
        
        return filepath
    
    def load_custom_dataset(self, file_path: str, 
                           text_column: str = 'text',
                           sentiment_column: str = 'sentiment') -> pd.DataFrame:
        """Load custom dataset from CSV file"""
        
        try:
            df = pd.read_csv(file_path)
            
            # Rename columns to standard format
            df = df.rename(columns={text_column: 'text', sentiment_column: 'sentiment'})
            
            # Clean the dataset
            df = self._clean_kaggle_dataset(df)
            
            self.logger.info(f"Loaded custom dataset: {len(df)} samples")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading custom dataset: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load Kaggle dataset (or generate synthetic if not available)
    df = loader.load_kaggle_product_reviews()
    
    # Get dataset statistics
    stats = loader.get_dataset_statistics(df)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create train/val/test splits
    train_df, val_df, test_df = loader.create_train_val_test_split(df)
    
    # Save processed datasets
    loader.save_processed_dataset(train_df, 'train_data.csv')
    loader.save_processed_dataset(val_df, 'val_data.csv')
    loader.save_processed_dataset(test_df, 'test_data.csv')