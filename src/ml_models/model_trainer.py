"""
Model Training and Retraining Script
"""
import argparse
import pandas as pd
import numpy as np
from sentiment_classifier import AdvancedSentimentClassifier
import os
import requests
import zipfile
from io import BytesIO

class ModelTrainer:
    """Handle model training and retraining operations"""
    
    def __init__(self):
        self.classifier = AdvancedSentimentClassifier()
        self.datasets = {
            'imdb': self._download_imdb_dataset,
            'sentiment140': self._download_sentiment140_dataset,
            'amazon': self._download_amazon_dataset,
            'synthetic': self._generate_synthetic_dataset
        }
    
    def _download_imdb_dataset(self):
        """Download and prepare IMDB movie reviews dataset"""
        print("Downloading IMDB dataset...")
        
        # For demonstration, we'll create a synthetic IMDB-like dataset
        # In production, you would download the actual IMDB dataset
        
        positive_reviews = [
            "This movie is absolutely fantastic! Great acting and amazing storyline.",
            "Brilliant film with outstanding performances. Highly recommended!",
            "Excellent movie with great direction and cinematography.",
            "Amazing story and wonderful characters. One of the best films ever!",
            "Fantastic movie with incredible acting and beautiful visuals.",
            "Outstanding film that exceeded all expectations. Must watch!",
            "Brilliant storytelling and exceptional performances throughout.",
            "Incredible movie with amazing plot twists and great ending.",
            "Excellent film with wonderful acting and beautiful cinematography.",
            "Amazing movie that kept me engaged from start to finish."
        ] * 50  # 500 positive reviews
        
        negative_reviews = [
            "Terrible movie with poor acting and boring storyline.",
            "Awful film that was a complete waste of time.",
            "Poor direction and terrible script. Very disappointing.",
            "Horrible movie with bad acting and confusing plot.",
            "Terrible film that I couldn't even finish watching.",
            "Awful movie with poor production quality and bad acting.",
            "Disappointing film with weak storyline and poor performances.",
            "Terrible movie that was boring and poorly executed.",
            "Awful film with bad direction and terrible acting.",
            "Poor movie that failed to meet any expectations."
        ] * 50  # 500 negative reviews
        
        neutral_reviews = [
            "Average movie with some good moments but nothing special.",
            "Okay film that was watchable but not memorable.",
            "Decent movie with mixed results. Some parts good, others not.",
            "Fair film that was neither great nor terrible.",
            "Standard movie that meets basic expectations.",
            "Mediocre film with average acting and storyline.",
            "Reasonable movie for casual viewing. Nothing extraordinary.",
            "Acceptable film that does the job but lacks excitement.",
            "So-so movie with average production quality.",
            "Regular film that's neither impressive nor disappointing."
        ] * 30  # 300 neutral reviews
        
        # Create DataFrame
        texts = positive_reviews + negative_reviews + neutral_reviews
        sentiments = (['positive'] * len(positive_reviews) + 
                     ['negative'] * len(negative_reviews) + 
                     ['neutral'] * len(neutral_reviews))
        
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        return df.sample(frac=1).reset_index(drop=True)
    
    def _download_sentiment140_dataset(self):
        """Download and prepare Sentiment140 dataset"""
        print("Downloading Sentiment140 dataset...")
        
        # Create synthetic Twitter-like dataset
        positive_tweets = [
            "Love this new product! Amazing quality and fast delivery #happy",
            "Best purchase ever! Highly recommend to everyone #awesome",
            "Fantastic service and great customer support! Very satisfied",
            "Amazing experience! Will definitely buy again #satisfied",
            "Excellent quality and perfect packaging! Love it so much",
            "Outstanding product that exceeded expectations! #recommend",
            "Great value for money and amazing features! Very happy",
            "Perfect product for my needs! Excellent quality #love",
            "Amazing customer service and fast shipping! Very pleased",
            "Fantastic product with great design! Highly satisfied"
        ] * 50  # 500 positive tweets
        
        negative_tweets = [
            "Terrible product! Poor quality and bad customer service #disappointed",
            "Worst purchase ever! Complete waste of money #awful",
            "Poor quality and doesn't work as advertised #frustrated",
            "Terrible experience! Would not recommend to anyone #bad",
            "Awful product that broke immediately! Very disappointed",
            "Poor customer service and defective product #terrible",
            "Horrible quality and overpriced! Avoid this product #waste",
            "Terrible experience with this company! Poor service #angry",
            "Awful product that doesn't meet expectations #disappointed",
            "Poor quality materials and bad design! Very unsatisfied"
        ] * 50  # 500 negative tweets
        
        neutral_tweets = [
            "Average product that does the job. Nothing special though",
            "Okay quality for the price. Could be better but acceptable",
            "Standard product with mixed results. Some good, some bad",
            "Fair quality and reasonable price. Nothing extraordinary",
            "Decent product that meets basic requirements adequately",
            "Mediocre quality with average performance. It's okay",
            "Reasonable product for this price range. Average overall",
            "Acceptable quality but could use some improvements",
            "So-so product with typical features. Nothing impressive",
            "Regular quality product. Neither great nor terrible"
        ] * 30  # 300 neutral tweets
        
        # Create DataFrame
        texts = positive_tweets + negative_tweets + neutral_tweets
        sentiments = (['positive'] * len(positive_tweets) + 
                     ['negative'] * len(negative_tweets) + 
                     ['neutral'] * len(neutral_tweets))
        
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        return df.sample(frac=1).reset_index(drop=True)
    
    def _download_amazon_dataset(self):
        """Download and prepare Amazon reviews dataset"""
        print("Downloading Amazon reviews dataset...")
        
        # Create synthetic Amazon-like reviews
        positive_reviews = [
            "Excellent product! Great quality and fast shipping. 5 stars!",
            "Amazing item that works perfectly. Highly recommend!",
            "Outstanding quality and great value for money. Love it!",
            "Perfect product that exceeded my expectations completely.",
            "Fantastic quality and excellent customer service. Very happy!",
            "Great product with amazing features. Will buy again!",
            "Excellent build quality and works as advertised. Satisfied!",
            "Amazing product that solved all my problems. Recommended!",
            "Outstanding value and great performance. Very pleased!",
            "Perfect item with excellent packaging. Five stars!"
        ] * 50  # 500 positive reviews
        
        negative_reviews = [
            "Terrible product! Poor quality and doesn't work properly.",
            "Awful item that broke after one day. Waste of money!",
            "Poor quality materials and bad design. Very disappointed.",
            "Horrible product that doesn't match the description at all.",
            "Terrible quality and overpriced. Would not recommend!",
            "Awful experience! Product arrived damaged and unusable.",
            "Poor construction and cheap materials. Complete waste!",
            "Terrible product that failed immediately. Very frustrated!",
            "Awful quality and poor customer service. Avoid this!",
            "Poor performance and doesn't work as advertised. Bad!"
        ] * 50  # 500 negative reviews
        
        neutral_reviews = [
            "Average product that does what it's supposed to do.",
            "Okay quality for the price. Nothing special but works.",
            "Decent item with some good and bad points. It's fine.",
            "Fair quality product that meets basic expectations.",
            "Standard item with average performance. Acceptable.",
            "Mediocre quality but gets the job done adequately.",
            "Reasonable product for this price range. It's okay.",
            "Acceptable quality with room for improvement.",
            "So-so product that works but could be better.",
            "Regular quality item. Neither impressive nor terrible."
        ] * 30  # 300 neutral reviews
        
        # Create DataFrame
        texts = positive_reviews + negative_reviews + neutral_reviews
        sentiments = (['positive'] * len(positive_reviews) + 
                     ['negative'] * len(negative_reviews) + 
                     ['neutral'] * len(neutral_reviews))
        
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        return df.sample(frac=1).reset_index(drop=True)
    
    def _generate_synthetic_dataset(self):
        """Generate synthetic dataset for training"""
        return self.classifier._generate_training_dataset()
    
    def train_new_model(self, dataset_name='synthetic', fine_tune_bert=False):
        """Train a new model on specified dataset"""
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"Training new model on {dataset_name} dataset...")
        
        # Load dataset
        df = self.datasets[dataset_name]()
        print(f"Loaded {len(df)} samples")
        
        # Display dataset statistics
        print("\nDataset statistics:")
        print(df['sentiment'].value_counts())
        
        # Train traditional ML models
        print("\nTraining traditional ML models...")
        scores = self.classifier.train_models(df)
        
        # Fine-tune BERT if requested
        if fine_tune_bert:
            print("\nFine-tuning BERT model...")
            self.classifier.fine_tune_bert(df)
        
        print("\nModel training completed!")
        return scores
    
    def retrain_model(self, new_data_path, combine_with_existing=True):
        """Retrain model with new data"""
        
        print("Retraining model with new data...")
        
        # Load new data
        if new_data_path.endswith('.csv'):
            new_df = pd.read_csv(new_data_path)
        else:
            raise ValueError("Only CSV files are supported")
        
        # Validate required columns
        required_columns = ['text', 'sentiment']
        if not all(col in new_df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")
        
        if combine_with_existing:
            # Load existing training data if available
            try:
                existing_df = self.classifier._generate_training_dataset()
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                print(f"Combined dataset size: {len(combined_df)}")
            except:
                combined_df = new_df
                print("Using only new data for retraining")
        else:
            combined_df = new_df
        
        # Retrain models
        scores = self.classifier.train_models(combined_df)
        
        print("Model retraining completed!")
        return scores
    
    def evaluate_model(self, test_data_path=None):
        """Evaluate trained model on test data"""
        
        if not self.classifier.best_model:
            if not self.classifier.load_model():
                raise ValueError("No trained model found. Please train a model first.")
        
        if test_data_path:
            # Load test data
            test_df = pd.read_csv(test_data_path)
            test_texts = test_df['text'].tolist()
            true_sentiments = test_df['sentiment'].tolist()
        else:
            # Use sample test data
            test_texts = [
                "This product is absolutely amazing! I love it so much.",
                "Terrible quality. Complete waste of money.",
                "It's okay, nothing special but does the job.",
                "Best purchase ever! Highly recommended!",
                "Awful product. Doesn't work at all."
            ]
            true_sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative']
        
        # Make predictions
        results = self.classifier.predict(test_texts)
        predicted_sentiments = [r['sentiment'] for r in results]
        
        # Calculate accuracy
        correct = sum(1 for true, pred in zip(true_sentiments, predicted_sentiments) if true == pred)
        accuracy = correct / len(true_sentiments)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        
        # Display detailed results
        print("\nDetailed Results:")
        for i, (text, true_sent, pred_sent, conf) in enumerate(zip(
            test_texts, true_sentiments, predicted_sentiments, 
            [r['confidence'] for r in results]
        )):
            status = "✓" if true_sent == pred_sent else "✗"
            print(f"{status} Text: {text[:50]}...")
            print(f"  True: {true_sent}, Predicted: {pred_sent} (Conf: {conf:.3f})")
            print()
        
        return accuracy, results
    
    def export_model_info(self):
        """Export model information and performance metrics"""
        
        if not self.classifier.best_model:
            if not self.classifier.load_model():
                raise ValueError("No trained model found.")
        
        info = {
            'model_name': getattr(self.classifier, 'best_model_name', 'unknown'),
            'classes': list(self.classifier.label_encoder.classes_),
            'model_type': type(self.classifier.best_model).__name__
        }
        
        print("Model Information:")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        return info

def main():
    """Main function for command-line interface"""
    
    parser = argparse.ArgumentParser(description='Train and manage sentiment analysis models')
    parser.add_argument('--action', choices=['train', 'retrain', 'evaluate'], 
                       required=True, help='Action to perform')
    parser.add_argument('--dataset', choices=['imdb', 'sentiment140', 'amazon', 'synthetic'],
                       default='synthetic', help='Dataset to use for training')
    parser.add_argument('--data-path', help='Path to custom dataset CSV file')
    parser.add_argument('--bert', action='store_true', help='Fine-tune BERT model')
    parser.add_argument('--combine', action='store_true', default=True,
                       help='Combine with existing data when retraining')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    if args.action == 'train':
        scores = trainer.train_new_model(args.dataset, args.bert)
        print("\nTraining completed!")
        
    elif args.action == 'retrain':
        if not args.data_path:
            raise ValueError("--data-path is required for retraining")
        scores = trainer.retrain_model(args.data_path, args.combine)
        print("\nRetraining completed!")
        
    elif args.action == 'evaluate':
        accuracy, results = trainer.evaluate_model(args.data_path)
        print(f"\nEvaluation completed! Accuracy: {accuracy:.4f}")
    
    # Export model info
    trainer.export_model_info()

if __name__ == "__main__":
    main()