"""
Advanced Sentiment Classification with Multiple ML Models
"""
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Deep Learning
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
import torch
from torch.utils.data import Dataset

# Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class SentimentDataset(Dataset):
    """Custom dataset for BERT fine-tuning"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class AdvancedSentimentClassifier:
    """Advanced sentiment classifier with multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model_dir = 'models'
        self.best_model = None
        self.best_vectorizer = None
        
        # Create models directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different ML models"""
        
        # Logistic Regression with different regularization
        self.models['logistic_l1'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', LogisticRegression(penalty='l1', solver='liblinear', random_state=42))
        ])
        
        self.models['logistic_l2'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words='english')),
            ('classifier', LogisticRegression(penalty='l2', random_state=42, max_iter=1000))
        ])
        
        # Support Vector Machine
        self.models['svm_linear'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', SVC(kernel='linear', random_state=42, probability=True))
        ])
        
        self.models['svm_rbf'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', SVC(kernel='rbf', random_state=42, probability=True))
        ])
        
        # Random Forest
        self.models['random_forest'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Naive Bayes
        self.models['naive_bayes'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        # Ensemble model
        self.models['ensemble'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                    ('nb', MultinomialNB())
                ],
                voting='soft'
            ))
        ])
    
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?.]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenization and lemmatization
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_dataset(self, dataset_path=None, dataset_type='imdb'):
        """Load and preprocess sentiment dataset"""
        
        if dataset_path and os.path.exists(dataset_path):
            # Load custom dataset
            df = pd.read_csv(dataset_path)
        else:
            # Generate synthetic dataset for demonstration
            df = self._generate_training_dataset()
        
        # Preprocess text
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Encode labels
        df['label_encoded'] = self.label_encoder.fit_transform(df['sentiment'])
        
        return df
    
    def _generate_training_dataset(self):
        """Generate synthetic training dataset"""
        
        # Positive examples
        positive_texts = [
            "This product is absolutely amazing! I love it so much and would definitely recommend it to everyone.",
            "Outstanding quality and excellent customer service. Exceeded all my expectations completely.",
            "Best purchase I've ever made! Works perfectly and arrived quickly. Five stars!",
            "Incredible value for money. High quality materials and fantastic design. Love it!",
            "Perfect product! Exactly what I needed. Great packaging and fast delivery.",
            "Excellent quality and works as advertised. Very satisfied with this purchase.",
            "Amazing product! Great features and easy to use. Highly recommended!",
            "Fantastic quality and excellent performance. Worth every penny spent.",
            "Love this product! Great design and functionality. Will buy again.",
            "Outstanding product with excellent build quality. Very impressed!",
            "Superb quality and great value. Exactly as described and works perfectly.",
            "Brilliant product! Easy to use and very effective. Highly recommend.",
            "Excellent purchase! High quality and great customer service.",
            "Amazing quality and fast shipping. Very happy with this product.",
            "Perfect product for my needs. Great quality and reasonable price."
        ] * 20  # 300 positive examples
        
        # Negative examples  
        negative_texts = [
            "Terrible product! Complete waste of money. Broke after just one day of use.",
            "Awful quality and poor customer service. Would not recommend to anyone.",
            "Worst purchase ever! Doesn't work as advertised. Returning immediately.",
            "Poor quality materials and terrible design. Very disappointed with this.",
            "Horrible product! Completely useless and overpriced. Avoid at all costs.",
            "Terrible quality and doesn't work properly. Waste of time and money.",
            "Awful product with poor build quality. Broke immediately after opening.",
            "Disappointing quality and terrible customer service. Very unsatisfied.",
            "Poor product that doesn't meet expectations. Cheap materials and bad design.",
            "Terrible experience! Product arrived damaged and doesn't work.",
            "Awful quality and overpriced. Not worth the money at all.",
            "Poor performance and terrible build quality. Very disappointed.",
            "Horrible product! Doesn't work as described. Complete waste of money.",
            "Terrible quality and poor functionality. Would not buy again.",
            "Awful product with many defects. Poor customer service too."
        ] * 20  # 300 negative examples
        
        # Neutral examples
        neutral_texts = [
            "Average product. Does the job but nothing special about it.",
            "Okay quality for the price. Has some good points and some bad.",
            "Decent product that works as expected. Nothing extraordinary though.",
            "Fair quality and reasonable price. Could be better but acceptable.",
            "Standard product that meets basic requirements. Nothing more.",
            "Mediocre quality with mixed results. Some features work well.",
            "Reasonable product for the price range. Average performance overall.",
            "Acceptable quality but could be improved. Does what it's supposed to.",
            "So-so product with average features. Neither great nor terrible.",
            "Regular quality product. Nothing to complain about or praise.",
            "Basic product that does the job adequately. Standard quality.",
            "Ordinary product with typical features. What you'd expect.",
            "Standard quality for this price range. Average performance.",
            "Moderate quality with some pros and cons. Acceptable overall.",
            "Typical product with average functionality. Nothing special."
        ] * 20  # 300 neutral examples
        
        # Create DataFrame
        texts = positive_texts + negative_texts + neutral_texts
        sentiments = ['positive'] * len(positive_texts) + ['negative'] * len(negative_texts) + ['neutral'] * len(neutral_texts)
        
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        # Shuffle the dataset
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df
    
    def train_models(self, df, test_size=0.2, cv_folds=5):
        """Train all models and find the best one"""
        
        X = df['cleaned_text']
        y = df['label_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        model_scores = {}
        
        print("Training multiple models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                
                # Test accuracy
                test_accuracy = model.score(X_test, y_test)
                
                model_scores[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'model': model
                }
                
                print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                print(f"Test Accuracy: {test_accuracy:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        # Find best model
        best_model_name = max(model_scores.keys(), 
                            key=lambda x: model_scores[x]['test_accuracy'])
        
        self.best_model = model_scores[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best test accuracy: {model_scores[best_model_name]['test_accuracy']:.4f}")
        
        # Detailed evaluation of best model
        y_pred = self.best_model.predict(X_test)
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save best model
        self.save_model()
        
        return model_scores
    
    def fine_tune_bert(self, df, model_name='distilbert-base-uncased', epochs=3):
        """Fine-tune BERT model for sentiment analysis"""
        
        print(f"Fine-tuning {model_name}...")
        
        # Prepare data
        X = df['cleaned_text'].tolist()
        y = df['label_encoded'].tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(self.label_encoder.classes_)
        )
        
        # Create datasets
        train_dataset = SentimentDataset(X_train, y_train, tokenizer)
        test_dataset = SentimentDataset(X_test, y_test, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'{self.model_dir}/bert_finetuned',
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{self.model_dir}/logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        
        # Train model
        trainer.train()
        
        # Save fine-tuned model
        trainer.save_model(f'{self.model_dir}/bert_finetuned')
        tokenizer.save_pretrained(f'{self.model_dir}/bert_finetuned')
        
        # Create pipeline for inference
        self.bert_pipeline = pipeline(
            "sentiment-analysis",
            model=f'{self.model_dir}/bert_finetuned',
            tokenizer=f'{self.model_dir}/bert_finetuned',
            return_all_scores=True
        )
        
        print("BERT fine-tuning completed!")
        
        return trainer
    
    def predict(self, texts, use_bert=False):
        """Predict sentiment for new texts"""
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        cleaned_texts = [self.preprocess_text(text) for text in texts]
        
        if use_bert and hasattr(self, 'bert_pipeline'):
            # Use fine-tuned BERT
            results = []
            for text in cleaned_texts:
                bert_result = self.bert_pipeline(text)[0]
                
                # Map BERT labels to our format
                label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                
                best_result = max(bert_result, key=lambda x: x['score'])
                sentiment = label_map.get(int(best_result['label'].split('_')[-1]), 'neutral')
                confidence = best_result['score']
                
                results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': confidence
                })
            
            return results
        
        elif self.best_model:
            # Use best traditional ML model
            predictions = self.best_model.predict(cleaned_texts)
            probabilities = self.best_model.predict_proba(cleaned_texts)
            
            results = []
            for i, text in enumerate(texts):
                sentiment = self.label_encoder.inverse_transform([predictions[i]])[0]
                confidence = np.max(probabilities[i])
                
                results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': confidence
                })
            
            return results
        
        else:
            raise ValueError("No trained model available. Please train a model first.")
    
    def save_model(self):
        """Save the best model and preprocessing components"""
        
        model_path = os.path.join(self.model_dir, 'best_sentiment_model.pkl')
        encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'classes': list(self.label_encoder.classes_),
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.model_dir, 'model_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load saved model and preprocessing components"""
        
        model_path = os.path.join(self.model_dir, 'best_sentiment_model.pkl')
        encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        metadata_path = os.path.join(self.model_dir, 'model_metadata.pkl')
        
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            with open(model_path, 'rb') as f:
                self.best_model = pickle.load(f)
            
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.best_model_name = metadata.get('model_name', 'unknown')
            
            print("Model loaded successfully!")
            return True
        
        return False
    
    def hyperparameter_tuning(self, df):
        """Perform hyperparameter tuning for best models"""
        
        X = df['cleaned_text']
        y = df['label_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Logistic Regression tuning
        lr_params = {
            'tfidf__max_features': [5000, 8000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2']
        }
        
        lr_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        print("Tuning Logistic Regression...")
        lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=3, scoring='accuracy', n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        
        print(f"Best LR params: {lr_grid.best_params_}")
        print(f"Best LR score: {lr_grid.best_score_:.4f}")
        
        # SVM tuning
        svm_params = {
            'tfidf__max_features': [5000, 8000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }
        
        svm_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', SVC(random_state=42, probability=True))
        ])
        
        print("Tuning SVM...")
        svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=3, scoring='accuracy', n_jobs=-1)
        svm_grid.fit(X_train, y_train)
        
        print(f"Best SVM params: {svm_grid.best_params_}")
        print(f"Best SVM score: {svm_grid.best_score_:.4f}")
        
        # Update best models
        self.models['tuned_logistic'] = lr_grid.best_estimator_
        self.models['tuned_svm'] = svm_grid.best_estimator_
        
        return lr_grid, svm_grid

if __name__ == "__main__":
    # Example usage
    classifier = AdvancedSentimentClassifier()
    
    # Load or generate dataset
    df = classifier.load_dataset()
    print(f"Dataset loaded: {len(df)} samples")
    
    # Train models
    scores = classifier.train_models(df)
    
    # Test predictions
    test_texts = [
        "This product is absolutely amazing!",
        "Terrible quality, waste of money.",
        "It's okay, nothing special."
    ]
    
    results = classifier.predict(test_texts)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
        print()