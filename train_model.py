#!/usr/bin/env python3
"""
Enhanced training script with Kaggle dataset support and overfitting prevention
Usage: python train_model.py --dataset kaggle --sample-size 50000 --detect-overfitting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml_models.model_trainer import ModelTrainer
import argparse
import json
from src.data_loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    """Enhanced training script with comprehensive options"""
    
    parser = argparse.ArgumentParser(
        description='Train sentiment analysis models with overfitting prevention',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on Kaggle dataset with overfitting detection
  python train_model.py --dataset kaggle --detect-overfitting
  
  # Train on custom dataset with grid search
  python train_model.py --dataset custom --data-path my_data.csv --grid-search
  
  # Quick training with synthetic data
  python train_model.py --dataset synthetic --sample-size 5000 --no-grid-search
  
  # Train specific models only
  python train_model.py --models logistic_regression random_forest
        """
    )
    
    # Dataset options
    parser.add_argument('--dataset', choices=['kaggle', 'custom', 'synthetic'],
                       default='kaggle', help='Dataset source to use')
    parser.add_argument('--data-path', help='Path to custom dataset CSV file (required for custom dataset)')
    parser.add_argument('--sample-size', type=int, help='Limit dataset size for faster training')
    
    # Training options
    parser.add_argument('--models', nargs='+', 
                       choices=['logistic_regression', 'svm_linear', 'random_forest', 'naive_bayes', 'all'],
                       default=['all'], help='Models to train')
    parser.add_argument('--grid-search', action='store_true', default=True,
                       help='Use grid search for hyperparameter tuning')
    parser.add_argument('--no-grid-search', action='store_true',
                       help='Disable grid search (faster training)')
    parser.add_argument('--detect-overfitting', action='store_true', default=True,
                       help='Perform overfitting analysis')
    parser.add_argument('--no-overfitting-detection', action='store_true',
                       help='Skip overfitting analysis (faster training)')
    
    # Output options
    parser.add_argument('--models-dir', default='models', help='Directory to save models')
    parser.add_argument('--data-dir', default='data', help='Directory for data files')
    parser.add_argument('--save-plots', action='store_true', help='Save overfitting analysis plots')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Handle argument conflicts
    if args.no_grid_search:
        args.grid_search = False
    if args.no_overfitting_detection:
        args.detect_overfitting = False
    
    # Validate arguments
    if args.dataset == 'custom' and not args.data_path:
        parser.error("--data-path is required when using custom dataset")
    
    print("🧠 SentimentFusions Pro - Enhanced Model Training")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Models: {args.models}")
    print(f"Grid Search: {args.grid_search}")
    print(f"Overfitting Detection: {args.detect_overfitting}")
    if args.sample_size:
        print(f"Sample Size: {args.sample_size}")
    print()
    
    # Initialize trainer
    trainer = ModelTrainer(models_dir=args.models_dir, data_dir=args.data_dir)
    
    # Filter models if specific ones requested
    if 'all' not in args.models:
        # Filter trainer's model configs to only include requested models
        filtered_configs = {name: config for name, config in trainer.model_configs.items() 
                          if name in args.models}
        trainer.model_configs = filtered_configs
        print(f"Training only: {list(filtered_configs.keys())}")
    
    try:
        # Start training
        print("🚀 Starting model training...")
        results = trainer.train_models(
            dataset_source=args.dataset,
            custom_file_path=args.data_path,
            sample_size=args.sample_size,
            use_grid_search=args.grid_search,
            detect_overfitting=args.detect_overfitting
        )
        
        print("\n✅ Training completed successfully!")
        
        # Display results summary
        print("\n📊 Model Performance Summary:")
        print("-" * 60)
        
        model_results = results.get('model_results', {})
        for model_name, metrics in model_results.items():
            if 'error' in metrics:
                print(f"❌ {model_name}: Error - {metrics['error']}")
                continue
                
            train_acc = metrics.get('train_accuracy', 0)
            val_acc = metrics.get('val_accuracy', 0)
            cv_mean = metrics.get('cv_mean', 0)
            cv_std = metrics.get('cv_std', 0)
            
            print(f"🤖 {model_name}:")
            print(f"   Train Accuracy: {train_acc:.4f}")
            print(f"   Val Accuracy:   {val_acc:.4f}")
            print(f"   CV Score:       {cv_mean:.4f} (±{cv_std:.4f})")
            
            # Overfitting analysis
            overfitting = metrics.get('overfitting_analysis', {})
            if overfitting and 'recommendation' in overfitting:
                print(f"   Overfitting:    {overfitting['recommendation']}")
            
            print()
        
        # Best model information
        best_model = results.get('best_model_name')
        best_score = results.get('best_model_score', 0)
        
        if best_model:
            print(f"🏆 Best Model: {best_model} (Val Accuracy: {best_score:.4f})")
        
        # Dataset information
        dataset_info = results.get('dataset_info', {})
        if dataset_info:
            print(f"\n📊 Dataset Information:")
            print(f"   Training samples: {dataset_info.get('train_size', 'unknown')}")
            print(f"   Validation samples: {dataset_info.get('val_size', 'unknown')}")
            print(f"   Test samples: {dataset_info.get('test_set_size', 'unknown')}")
            print(f"   Features: {dataset_info.get('feature_count', 'unknown')}")
        
        # Save detailed results
        if args.verbose:
            results_file = os.path.join(args.models_dir, 'latest_training_results.json')
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if key == 'model_results':
                    json_results[key] = {}
                    for model_name, model_data in value.items():
                        json_results[key][model_name] = {}
                        for k, v in model_data.items():
                            if k == 'model':
                                continue  # Skip model object
                            elif isinstance(v, np.ndarray):
                                json_results[key][model_name][k] = v.tolist()
                            else:
                                json_results[key][model_name][k] = v
                else:
                    json_results[key] = value
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Generate overfitting plots
        if args.save_plots and args.detect_overfitting:
            print("\n📈 Generating overfitting analysis plots...")
            generate_overfitting_plots(results, args.models_dir)
        
        # Training recommendations
        print("\n💡 Training Recommendations:")
        generate_training_recommendations(results)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

def generate_overfitting_plots(results: dict, models_dir: str):
    """Generate and save overfitting analysis plots"""
    
    model_results = results.get('model_results', {})
    overfitting_analyses = results.get('overfitting_analyses', {})
    
    if not overfitting_analyses:
        print("   No overfitting analysis data available")
        return
    
    # Create plots directory
    plots_dir = os.path.join(models_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for model_name, analysis in overfitting_analyses.items():
        if 'error' in analysis or not analysis:
            continue
            
        try:
            # Validation curve plot
            param_range = analysis.get('param_range', [])
            train_scores = analysis.get('train_scores_mean', [])
            val_scores = analysis.get('val_scores_mean', [])
            train_std = analysis.get('train_scores_std', [])
            val_std = analysis.get('val_scores_std', [])
            
            if not all([param_range, train_scores, val_scores]):
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Convert param_range to strings if it contains None
            param_labels = [str(p) if p is not None else 'None' for p in param_range]
            x_pos = range(len(param_labels))
            
            plt.plot(x_pos, train_scores, 'o-', color='blue', label='Training Score')
            plt.fill_between(x_pos, 
                           np.array(train_scores) - np.array(train_std),
                           np.array(train_scores) + np.array(train_std),
                           alpha=0.1, color='blue')
            
            plt.plot(x_pos, val_scores, 'o-', color='red', label='Validation Score')
            plt.fill_between(x_pos,
                           np.array(val_scores) - np.array(val_std),
                           np.array(val_scores) + np.array(val_std),
                           alpha=0.1, color='red')
            
            plt.xlabel(f"{analysis.get('param_name', 'Parameter')}")
            plt.ylabel('Accuracy Score')
            plt.title(f'Validation Curve - {model_name}')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.xticks(x_pos, param_labels, rotation=45)
            
            # Add overfitting indicator
            if analysis.get('is_overfitting'):
                plt.text(0.02, 0.98, 'Overfitting Detected', 
                        transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                        verticalalignment='top', color='white', fontweight='bold')
            elif analysis.get('is_underfitting'):
                plt.text(0.02, 0.98, 'Underfitting Detected',
                        transform=plt.gca().transAxes,
                        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7),
                        verticalalignment='top', color='white', fontweight='bold')
            else:
                plt.text(0.02, 0.98, 'Good Fit',
                        transform=plt.gca().transAxes,
                        bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                        verticalalignment='top', color='white', fontweight='bold')
            
            plt.tight_layout()
            
            plot_path = os.path.join(plots_dir, f'{model_name}_validation_curve.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   Saved plot: {plot_path}")
            
        except Exception as e:
            print(f"   Error generating plot for {model_name}: {e}")

def generate_training_recommendations(results: dict):
    """Generate training recommendations based on results"""
    
    model_results = results.get('model_results', {})
    overfitting_analyses = results.get('overfitting_analyses', {})
    
    recommendations = []
    
    # Check for overfitting issues
    overfitting_models = []
    underfitting_models = []
    
    for model_name, analysis in overfitting_analyses.items():
        if analysis.get('is_overfitting'):
            overfitting_models.append(model_name)
        elif analysis.get('is_underfitting'):
            underfitting_models.append(model_name)
    
    if overfitting_models:
        recommendations.append(f"🔴 Overfitting detected in: {', '.join(overfitting_models)}")
        recommendations.append("   → Consider: More data, stronger regularization, simpler models")
    
    if underfitting_models:
        recommendations.append(f"🟡 Underfitting detected in: {', '.join(underfitting_models)}")
        recommendations.append("   → Consider: More complex models, better features, less regularization")
    
    # Check performance gaps
    large_gaps = []
    for model_name, metrics in model_results.items():
        if 'error' in metrics:
            continue
        train_acc = metrics.get('train_accuracy', 0)
        val_acc = metrics.get('val_accuracy', 0)
        gap = train_acc - val_acc
        
        if gap > 0.1:  # 10% gap
            large_gaps.append((model_name, gap))
    
    if large_gaps:
        gap_models = [f"{name} ({gap:.3f})" for name, gap in large_gaps]
        recommendations.append(f"⚠️  Large train-val gaps: {', '.join(gap_models)}")
        recommendations.append("   → Consider: Cross-validation, regularization, more validation data")
    
    # Performance recommendations
    best_model = results.get('best_model_name')
    best_score = results.get('best_model_score', 0)
    
    if best_score < 0.7:
        recommendations.append("📈 Low overall performance detected")
        recommendations.append("   → Consider: Better features, more data, ensemble methods")
    elif best_score > 0.9:
        recommendations.append("🎯 Excellent performance achieved!")
        recommendations.append("   → Consider: Final testing, production deployment")
    
    # Dataset recommendations
    dataset_info = results.get('dataset_info', {})
    train_size = dataset_info.get('train_size', 0)
    
    if train_size < 1000:
        recommendations.append("📊 Small dataset detected")
        recommendations.append("   → Consider: Data augmentation, transfer learning, more data collection")
    
    # Print recommendations
    if recommendations:
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print("   ✅ No specific recommendations - training looks good!")

if __name__ == "__main__":
    exit(main())