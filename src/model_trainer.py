@@ .. @@
 """
 Advanced Model Training with Overfitting Prevention
 """
 import pandas as pd
 import numpy as np
 import pickle
 import joblib
 import os
 from datetime import datetime
 from typing import Dict, List, Tuple, Any, Optional
 import warnings
 warnings.filterwarnings('ignore')
 
 # ML Libraries
 from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.linear_model import LogisticRegression
 from sklearn.svm import SVC
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.naive_bayes import MultinomialNB
 from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 from sklearn.pipeline import Pipeline
 from sklearn.preprocessing import LabelEncoder
+from sklearn.model_selection import validation_curve, learning_curve
+from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
 
 # Feature Engineering
 from .feature_engineering import FeatureEngineer
 from .preprocessing import TextPreprocessor
+from .data_loader import DataLoader
 
 import logging
 
 class ModelTrainer:
     """Advanced model trainer with overfitting prevention"""
     
-    def __init__(self, models_dir: str = 'models'):
+    def __init__(self, models_dir: str = 'models', data_dir: str = 'data'):
         self.models_dir = models_dir
+        self.data_dir = data_dir
         self.preprocessor = TextPreprocessor()
         self.feature_engineer = FeatureEngineer()
+        self.data_loader = DataLoader(data_dir)
         self.label_encoder = LabelEncoder()
         self.logger = self._setup_logger()
         
         # Create directories
         os.makedirs(models_dir, exist_ok=True)
+        os.makedirs(data_dir, exist_ok=True)
         
         # Model configurations with regularization
         self.model_configs = {
             'logistic_regression': {
                 'model': LogisticRegression(
-                    random_state=42, 
-                    max_iter=1000,
-                    solver='liblinear'
+                    random_state=42,
+                    max_iter=2000,
+                    solver='liblinear',
+                    penalty='l2',  # L2 regularization
+                    C=1.0  # Regularization strength
                 ),
                 'param_grid': {
-                    'C': [0.1, 1, 10],
-                    'penalty': ['l1', 'l2']
+                    'C': [0.01, 0.1, 1, 10, 100],  # More regularization options
+                    'penalty': ['l1', 'l2'],
+                    'solver': ['liblinear', 'saga']
                 }
             },
             'svm_linear': {
                 'model': SVC(
                     kernel='linear',
                     random_state=42,
-                    probability=True
+                    probability=True,
+                    C=1.0  # Regularization parameter
                 ),
                 'param_grid': {
-                    'C': [0.1, 1, 10],
-                    'gamma': ['scale', 'auto']
+                    'C': [0.01, 0.1, 1, 10, 100],  # More regularization options
+                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                 }
             },
             'random_forest': {
                 'model': RandomForestClassifier(
                     n_estimators=100,
                     random_state=42,
-                    n_jobs=-1
+                    n_jobs=-1,
+                    max_depth=10,  # Prevent overfitting
+                    min_samples_split=5,  # Prevent overfitting
+                    min_samples_leaf=2   # Prevent overfitting
                 ),
                 'param_grid': {
-                    'n_estimators': [50, 100, 200],
-                    'max_depth': [None, 10, 20],
-                    'min_samples_split': [2, 5, 10]
+                    'n_estimators': [50, 100, 200, 300],
+                    'max_depth': [5, 10, 15, 20, None],
+                    'min_samples_split': [2, 5, 10, 15],
+                    'min_samples_leaf': [1, 2, 4, 6]
                 }
             },
             'naive_bayes': {
                 'model': MultinomialNB(
-                    alpha=1.0
+                    alpha=1.0  # Laplace smoothing
                 ),
                 'param_grid': {
-                    'alpha': [0.1, 1.0, 10.0]
+                    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]  # More smoothing options
                 }
             }
         }
         
+        # Cross-validation configuration
+        self.cv_config = {
+            'cv_folds': 5,
+            'scoring': 'accuracy',
+            'n_jobs': -1
+        }
+        
+        # Early stopping configuration
+        self.early_stopping_config = {
+            'patience': 5,
+            'min_delta': 0.001,
+            'monitor': 'val_accuracy'
+        }
+        
     def _setup_logger(self) -> logging.Logger:
         """Setup logging for training operations"""
         logger = logging.getLogger('ModelTrainer')
         logger.setLevel(logging.INFO)
         
         if not logger.handlers:
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             logger.addHandler(handler)
         
         return logger
     
-    def train_models(self, df: pd.DataFrame, 
+    def load_and_prepare_data(self, dataset_source: str = 'kaggle',
+                             custom_file_path: Optional[str] = None,
+                             sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
+        """
+        Load and prepare data with proper train/val/test splits
+        
+        Args:
+            dataset_source: 'kaggle', 'custom', or 'synthetic'
+            custom_file_path: Path to custom dataset file
+            sample_size: Limit dataset size for faster training
+            
+        Returns:
+            Tuple of (train_df, val_df, test_df)
+        """
+        
+        self.logger.info(f"Loading dataset from source: {dataset_source}")
+        
+        if dataset_source == 'kaggle':
+            df = self.data_loader.load_kaggle_product_reviews(custom_file_path)
+        elif dataset_source == 'custom' and custom_file_path:
+            df = self.data_loader.load_custom_dataset(custom_file_path)
+        else:
+            # Generate synthetic data
+            df = self.data_loader._generate_synthetic_kaggle_like_data(
+                n_samples=sample_size or 10000
+            )
+        
+        # Sample data if requested
+        if sample_size and len(df) > sample_size:
+            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
+            self.logger.info(f"Sampled dataset to {sample_size} examples")
+        
+        # Get dataset statistics
+        stats = self.data_loader.get_dataset_statistics(df)
+        self.logger.info("Dataset Statistics:")
+        for key, value in stats.items():
+            self.logger.info(f"  {key}: {value}")
+        
+        # Create train/val/test splits
+        train_df, val_df, test_df = self.data_loader.create_train_val_test_split(
+            df, train_size=0.7, val_size=0.15, test_size=0.15
+        )
+        
+        return train_df, val_df, test_df
+    
+    def detect_overfitting_underfitting(self, model, X_train, y_train, X_val, y_val,
+                                      param_name: str = 'C', param_range: List = None) -> Dict[str, Any]:
+        """
+        Detect overfitting/underfitting using validation curves
+        
+        Args:
+            model: Sklearn model
+            X_train, y_train: Training data
+            X_val, y_val: Validation data
+            param_name: Parameter to vary for validation curve
+            param_range: Range of parameter values to test
+            
+        Returns:
+            Dictionary with overfitting analysis results
+        """
+        
+        if param_range is None:
+            if param_name == 'C':
+                param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
+            elif param_name == 'max_depth':
+                param_range = [3, 5, 7, 10, 15, 20, None]
+            elif param_name == 'n_estimators':
+                param_range = [10, 50, 100, 200, 300, 500]
+            else:
+                param_range = [0.1, 1, 10]
+        
+        try:
+            # Generate validation curve
+            train_scores, val_scores = validation_curve(
+                model, X_train, y_train,
+                param_name=param_name,
+                param_range=param_range,
+                cv=3,  # Reduced CV for speed
+                scoring='accuracy',
+                n_jobs=-1
+            )
+            
+            train_mean = np.mean(train_scores, axis=1)
+            train_std = np.std(train_scores, axis=1)
+            val_mean = np.mean(val_scores, axis=1)
+            val_std = np.std(val_scores, axis=1)
+            
+            # Find best parameter
+            best_idx = np.argmax(val_mean)
+            best_param = param_range[best_idx]
+            
+            # Detect overfitting (large gap between train and val scores)
+            score_gaps = train_mean - val_mean
+            max_gap = np.max(score_gaps)
+            avg_gap = np.mean(score_gaps)
+            
+            # Overfitting indicators
+            is_overfitting = max_gap > 0.1 or avg_gap > 0.05
+            is_underfitting = np.max(val_mean) < 0.7  # Low overall performance
+            
+            analysis = {
+                'param_name': param_name,
+                'param_range': param_range,
+                'train_scores_mean': train_mean.tolist(),
+                'val_scores_mean': val_mean.tolist(),
+                'train_scores_std': train_std.tolist(),
+                'val_scores_std': val_std.tolist(),
+                'best_param': best_param,
+                'best_val_score': val_mean[best_idx],
+                'score_gaps': score_gaps.tolist(),
+                'max_gap': max_gap,
+                'avg_gap': avg_gap,
+                'is_overfitting': is_overfitting,
+                'is_underfitting': is_underfitting,
+                'recommendation': self._get_overfitting_recommendation(
+                    is_overfitting, is_underfitting, max_gap, np.max(val_mean)
+                )
+            }
+            
+            return analysis
+            
+        except Exception as e:
+            self.logger.warning(f"Could not generate validation curve: {str(e)}")
+            return {'error': str(e)}
+    
+    def _get_overfitting_recommendation(self, is_overfitting: bool, is_underfitting: bool,
+                                      max_gap: float, best_val_score: float) -> str:
+        """Get recommendation based on overfitting analysis"""
+        
+        if is_overfitting and max_gap > 0.15:
+            return "Strong overfitting detected. Increase regularization, reduce model complexity, or get more data."
+        elif is_overfitting:
+            return "Mild overfitting detected. Consider slight regularization increase."
+        elif is_underfitting and best_val_score < 0.6:
+            return "Strong underfitting detected. Increase model complexity or improve features."
+        elif is_underfitting:
+            return "Mild underfitting detected. Consider increasing model complexity."
+        else:
+            return "Model appears well-balanced. Good fit achieved."
+    
+    def train_models_with_validation(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                      test_size: float = 0.2,
-                     cv_folds: int = 5,
-                     use_grid_search: bool = True) -> Dict[str, Any]:
+                     use_grid_search: bool = True,
+                     detect_overfitting: bool = True) -> Dict[str, Any]:
         """
-        Train multiple models with cross-validation and hyperparameter tuning
+        Train multiple models with proper validation and overfitting detection
         
         Args:
-            df: Input dataframe with 'text' and 'sentiment' columns
+            train_df: Training dataframe
+            val_df: Validation dataframe
             test_size: Proportion of data for testing
-            cv_folds: Number of cross-validation folds
             use_grid_search: Whether to use grid search for hyperparameter tuning
+            detect_overfitting: Whether to perform overfitting analysis
             
         Returns:
             Dictionary with model performance results
         """
         
-        self.logger.info("Starting model training with cross-validation...")
+        self.logger.info("Starting model training with validation...")
         
         # Preprocess text data
-        self.logger.info("Preprocessing text data...")
-        df['cleaned_text'] = df['text'].apply(self.preprocessor.preprocess_text)
+        self.logger.info("Preprocessing training data...")
+        train_df['cleaned_text'] = train_df['text'].apply(self.preprocessor.preprocess_text)
+        val_df['cleaned_text'] = val_df['text'].apply(self.preprocessor.preprocess_text)
         
         # Remove empty texts
-        df = df[df['cleaned_text'].str.len() > 0]
+        train_df = train_df[train_df['cleaned_text'].str.len() > 0]
+        val_df = val_df[val_df['cleaned_text'].str.len() > 0]
         
         # Encode labels
-        y = self.label_encoder.fit_transform(df['sentiment'])
-        X = df['cleaned_text']
+        y_train = self.label_encoder.fit_transform(train_df['sentiment'])
+        y_val = self.label_encoder.transform(val_df['sentiment'])
+        X_train = train_df['cleaned_text']
+        X_val = val_df['cleaned_text']
         
-        # Split data
-        X_train, X_test, y_train, y_test = train_test_split(
-            X, y, test_size=test_size, random_state=42, stratify=y
-        )
+        self.logger.info(f"Training set: {len(X_train)} samples")
+        self.logger.info(f"Validation set: {len(X_val)} samples")
         
         # Feature engineering
         self.logger.info("Extracting features...")
-        X_train_features = self.feature_engineer.extract_features(X_train, fit=True)
-        X_test_features = self.feature_engineer.extract_features(X_test, fit=False)
+        X_train_features = self.feature_engineer.extract_features(X_train, fit=True)
+        X_val_features = self.feature_engineer.extract_features(X_val, fit=False)
         
         model_results = {}
         best_model = None
         best_score = 0
         best_model_name = None
+        overfitting_analyses = {}
         
         # Train each model
         for model_name, config in self.model_configs.items():
             self.logger.info(f"\nTraining {model_name}...")
             
             try:
                 if use_grid_search and 'param_grid' in config:
                     # Grid search with cross-validation
                     self.logger.info(f"Performing grid search for {model_name}...")
                     
-                    grid_search = GridSearchCV(
+                    # Use StratifiedKFold for better validation
+                    cv = StratifiedKFold(n_splits=self.cv_config['cv_folds'], 
+                                       shuffle=True, random_state=42)
+                    
+                    grid_search = GridSearchCV(
                         config['model'],
                         config['param_grid'],
-                        cv=cv_folds,
-                        scoring='accuracy',
-                        n_jobs=-1,
+                        cv=cv,
+                        scoring=self.cv_config['scoring'],
+                        n_jobs=self.cv_config['n_jobs'],
                         verbose=1
                     )
                     
                     grid_search.fit(X_train_features, y_train)
                     model = grid_search.best_estimator_
                     best_params = grid_search.best_params_
                     
                     self.logger.info(f"Best parameters for {model_name}: {best_params}")
                     
                 else:
                     # Train with default parameters
                     model = config['model']
                     model.fit(X_train_features, y_train)
                     best_params = {}
                 
                 # Evaluate model
-                train_score = model.score(X_train_features, y_train)
-                test_score = model.score(X_test_features, y_test)
+                train_score = model.score(X_train_features, y_train)
+                val_score = model.score(X_val_features, y_val)
                 
                 # Cross-validation score
-                cv_scores = cross_val_score(model, X_train_features, y_train, 
-                                          cv=cv_folds, scoring='accuracy')
+                cv = StratifiedKFold(n_splits=self.cv_config['cv_folds'], 
+                                   shuffle=True, random_state=42)
+                cv_scores = cross_val_score(model, X_train_features, y_train,
+                                          cv=cv, scoring=self.cv_config['scoring'])
                 
                 # Predictions for detailed metrics
-                y_pred = model.predict(X_test_features)
-                y_pred_proba = model.predict_proba(X_test_features) if hasattr(model, 'predict_proba') else None
+                y_val_pred = model.predict(X_val_features)
+                y_val_pred_proba = model.predict_proba(X_val_features) if hasattr(model, 'predict_proba') else None
                 
                 # Classification report
-                class_report = classification_report(y_test, y_pred, 
+                class_report = classification_report(y_val, y_val_pred,
                                                    target_names=self.label_encoder.classes_,
                                                    output_dict=True)
                 
                 # Confusion matrix
-                conf_matrix = confusion_matrix(y_test, y_pred)
+                conf_matrix = confusion_matrix(y_val, y_val_pred)
+                
+                # Overfitting analysis
+                overfitting_analysis = {}
+                if detect_overfitting:
+                    self.logger.info(f"Analyzing overfitting for {model_name}...")
+                    
+                    # Choose appropriate parameter for validation curve
+                    if model_name == 'logistic_regression':
+                        overfitting_analysis = self.detect_overfitting_underfitting(
+                            config['model'], X_train_features, y_train, X_val_features, y_val,
+                            param_name='C', param_range=[0.001, 0.01, 0.1, 1, 10, 100]
+                        )
+                    elif model_name == 'svm_linear':
+                        overfitting_analysis = self.detect_overfitting_underfitting(
+                            config['model'], X_train_features, y_train, X_val_features, y_val,
+                            param_name='C', param_range=[0.001, 0.01, 0.1, 1, 10, 100]
+                        )
+                    elif model_name == 'random_forest':
+                        overfitting_analysis = self.detect_overfitting_underfitting(
+                            config['model'], X_train_features, y_train, X_val_features, y_val,
+                            param_name='max_depth', param_range=[3, 5, 10, 15, 20, None]
+                        )
+                    elif model_name == 'naive_bayes':
+                        overfitting_analysis = self.detect_overfitting_underfitting(
+                            config['model'], X_train_features, y_train, X_val_features, y_val,
+                            param_name='alpha', param_range=[0.01, 0.1, 1, 10]
+                        )
+                
+                overfitting_analyses[model_name] = overfitting_analysis
                 
                 # Store results
                 model_results[model_name] = {
                     'model': model,
                     'train_accuracy': train_score,
-                    'test_accuracy': test_score,
+                    'val_accuracy': val_score,
                     'cv_mean': cv_scores.mean(),
                     'cv_std': cv_scores.std(),
                     'best_params': best_params,
                     'classification_report': class_report,
                     'confusion_matrix': conf_matrix.tolist(),
-                    'predictions': y_pred.tolist(),
-                    'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
+                    'predictions': y_val_pred.tolist(),
+                    'prediction_probabilities': y_val_pred_proba.tolist() if y_val_pred_proba is not None else None,
+                    'overfitting_analysis': overfitting_analysis
                 }
                 
                 # Track best model
-                if test_score > best_score:
-                    best_score = test_score
+                if val_score > best_score:
+                    best_score = val_score
                     best_model = model
                     best_model_name = model_name
                 
                 # Log results
                 self.logger.info(f"{model_name} Results:")
                 self.logger.info(f"  Train Accuracy: {train_score:.4f}")
-                self.logger.info(f"  Test Accuracy: {test_score:.4f}")
+                self.logger.info(f"  Val Accuracy: {val_score:.4f}")
                 self.logger.info(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
+                
+                # Log overfitting analysis
+                if overfitting_analysis and 'recommendation' in overfitting_analysis:
+                    self.logger.info(f"  Overfitting Analysis: {overfitting_analysis['recommendation']}")
                 
             except Exception as e:
                 self.logger.error(f"Error training {model_name}: {str(e)}")
                 model_results[model_name] = {'error': str(e)}
         
         # Save best model
         if best_model is not None:
             self.save_model(best_model, best_model_name)
             self.logger.info(f"\nBest model: {best_model_name} (Accuracy: {best_score:.4f})")
         
         # Compile final results
         final_results = {
             'model_results': model_results,
             'best_model_name': best_model_name,
             'best_model_score': best_score,
             'label_encoder_classes': self.label_encoder.classes_.tolist(),
-            'training_timestamp': datetime.now().isoformat()
+            'training_timestamp': datetime.now().isoformat(),
+            'overfitting_analyses': overfitting_analyses,
+            'dataset_info': {
+                'train_size': len(train_df),
+                'val_size': len(val_df),
+                'feature_count': X_train_features.shape[1] if hasattr(X_train_features, 'shape') else 'unknown'
+            }
         }
         
         # Save training results
         self.save_training_results(final_results)
         
         return final_results
     
+    def train_models(self, dataset_source: str = 'kaggle',
+                    custom_file_path: Optional[str] = None,
+                    sample_size: Optional[int] = None,
+                    use_grid_search: bool = True,
+                    detect_overfitting: bool = True) -> Dict[str, Any]:
+        """
+        Complete training pipeline with data loading and validation
+        
+        Args:
+            dataset_source: 'kaggle', 'custom', or 'synthetic'
+            custom_file_path: Path to custom dataset file
+            sample_size: Limit dataset size for faster training
+            use_grid_search: Whether to use grid search
+            detect_overfitting: Whether to perform overfitting analysis
+            
+        Returns:
+            Dictionary with complete training results
+        """
+        
+        # Load and prepare data
+        train_df, val_df, test_df = self.load_and_prepare_data(
+            dataset_source, custom_file_path, sample_size
+        )
+        
+        # Train models with validation
+        results = self.train_models_with_validation(
+            train_df, val_df, use_grid_search=use_grid_search,
+            detect_overfitting=detect_overfitting
+        )
+        
+        # Add test set information
+        results['test_set_size'] = len(test_df)
+        
+        # Save test set for later evaluation
+        test_df.to_csv(os.path.join(self.data_dir, 'test_set.csv'), index=False)
+        
+        return results
+    
     def save_model(self, model, model_name: str):
         """Save trained model and associated components"""
         
         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
         
         # Save model
         model_path = os.path.join(self.models_dir, f'{model_name}_{timestamp}.pkl')
         joblib.dump(model, model_path)
         
         # Save feature engineer
         feature_path = os.path.join(self.models_dir, f'feature_engineer_{timestamp}.pkl')
         joblib.dump(self.feature_engineer, feature_path)
         
         # Save label encoder
         encoder_path = os.path.join(self.models_dir, f'label_encoder_{timestamp}.pkl')
         joblib.dump(self.label_encoder, encoder_path)
         
         # Save model metadata
         metadata = {
             'model_name': model_name,
             'model_path': model_path,
             'feature_engineer_path': feature_path,
             'label_encoder_path': encoder_path,
             'timestamp': timestamp,
             'class_names': self.label_encoder.classes_.tolist()
         }
         
         metadata_path = os.path.join(self.models_dir, f'model_metadata_{timestamp}.json')
         import json
         with open(metadata_path, 'w') as f:
             json.dump(metadata, f, indent=2)
         
         self.logger.info(f"Model saved: {model_path}")
         
         return model_path
     
     def save_training_results(self, results: Dict[str, Any]):
         """Save comprehensive training results"""
         
         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
         results_path = os.path.join(self.models_dir, f'training_results_{timestamp}.pkl')
         
         with open(results_path, 'wb') as f:
             pickle.dump(results, f)
         
         self.logger.info(f"Training results saved: {results_path}")
         
         return results_path