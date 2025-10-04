#!/usr/bin/env python3
"""
Large-scale training script for fake news detection.
Optimized for datasets with 40k+ articles.
"""
import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import gc

warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.large_scale_preprocess import LargeScaleNewsPreprocessor, load_large_dataset
from src.model import FakeNewsClassifier

def train_large_scale(data_path='data/large_scale_fake_news_dataset.csv', 
                     sample_size=None, 
                     max_features=20000,
                     model_dir='models_large_scale'):
    """
    Train fake news detector on large dataset.
    
    Args:
        data_path (str): Path to dataset
        sample_size (int): Optional sample size for testing
        max_features (int): Number of TF-IDF features
        model_dir (str): Directory to save models
    """
    
    print("🚀" * 20)
    print("  LARGE-SCALE FAKE NEWS DETECTOR TRAINING")
    print("🚀" * 20)
    
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load dataset
    print(f"\n📁 Loading dataset from {data_path}...")
    df = load_large_dataset(data_path)
    
    if df is None:
        print("❌ Failed to load dataset")
        return None
    
    # Sample if specified
    if sample_size and sample_size < len(df):
        print(f"\n🎯 Sampling {sample_size:,} articles...")
        # Use sklearn's train_test_split for stratified sampling
        df_sample, _ = train_test_split(
            df, train_size=sample_size, random_state=42, stratify=df['label']
        )
        df = df_sample
    
    # Print statistics
    real_count = len(df[df['label'] == 0])
    fake_count = len(df[df['label'] == 1])
    
    print(f"\n📊 Dataset statistics:")
    print(f"   Total articles: {len(df):,}")
    print(f"   Real news: {real_count:,} ({real_count/len(df)*100:.1f}%)")
    print(f"   Fake news: {fake_count:,} ({fake_count/len(df)*100:.1f}%)")
    
    # Step 2: Initialize preprocessor
    print(f"\n🔧 Initializing preprocessor (max_features={max_features:,})...")
    preprocessor = LargeScaleNewsPreprocessor(
        max_features=max_features,
        use_lemmatization=True,
        batch_size=1000
    )
    
    # Step 3: Preprocess data
    print(f"\n🔄 Starting preprocessing pipeline...")
    texts, labels = preprocessor.prepare_data_large_scale(df)
    
    # Step 4: Split data
    print(f"\n🔄 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Testing set: {len(X_test):,} samples")
    
    # Step 5: Vectorization
    print(f"\n🔧 Fitting TF-IDF vectorizer...")
    preprocessor.fit_vectorizer_large_scale(X_train)
    
    print(f"🔄 Transforming training data...")
    X_train_tfidf = preprocessor.transform_texts_large_scale(X_train)
    
    print(f"🔄 Transforming testing data...")
    X_test_tfidf = preprocessor.transform_texts_large_scale(X_test)
    
    # Memory cleanup
    del X_train, X_test, texts, labels, df
    gc.collect()
    
    # Step 6: Train models
    print(f"\n🎯 Training all models...")
    classifier = FakeNewsClassifier()
    
    results = classifier.train_all_models(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    
    # Step 7: Enhanced evaluation
    print(f"\n📈 Enhanced Model Evaluation:")
    print("=" * 60)
    
    enhanced_results = {}
    
    for model_name, model_results in results.items():
        print(f"\n🤖 {model_name.upper()} Results:")
        
        # Get predictions
        model = classifier.trained_models[model_name]
        y_pred = model.predict(X_test_tfidf)
        
        # Calculate AUC if possible
        try:
            y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = None
        
        enhanced_results[model_name] = {
            **model_results,
            'auc_score': auc_score
        }
        
        # Print metrics
        print(f"   Training Accuracy: {model_results['train_accuracy']:.4f}")
        print(f"   CV Score: {model_results['cv_mean']:.4f} (±{model_results['cv_std']:.4f})")
        print(f"   Test Accuracy: {model_results['test_accuracy']:.4f}")
        if auc_score:
            print(f"   AUC Score: {auc_score:.4f}")
        
        print("-" * 50)
    
    # Step 8: Save models and results
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n💾 Saving models to {model_dir}/...")
    
    # Save vectorizer
    vectorizer_path = os.path.join(model_dir, 'large_scale_vectorizer.pkl')
    preprocessor.save_vectorizer(vectorizer_path)
    
    # Save best model
    best_model_path = os.path.join(model_dir, f'best_model_large_scale_{classifier.best_model_name}.pkl')
    classifier.save_model(classifier.best_model_name, best_model_path)
    
    # Save all models
    for model_name in classifier.trained_models.keys():
        model_path = os.path.join(model_dir, f'{model_name}_large_scale.pkl')
        classifier.save_model(model_name, model_path)
    
    # Save results
    results_data = []
    for model_name, results in enhanced_results.items():
        results_data.append({
            'Model': model_name,
            'Train_Accuracy': results.get('train_accuracy', 0),
            'CV_Mean': results.get('cv_mean', 0),
            'CV_Std': results.get('cv_std', 0),
            'Test_Accuracy': results.get('test_accuracy', 0),
            'AUC_Score': results.get('auc_score', None)
        })
    
    results_df = pd.DataFrame(results_data)
    results_path = os.path.join(model_dir, 'large_scale_training_results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n🎉" * 20)
    print("  TRAINING COMPLETED SUCCESSFULLY!")
    print("🎉" * 20)
    print(f"Duration: {duration}")
    print(f"Best Model: {classifier.best_model_name}")
    print(f"Best CV Score: {max([r.get('cv_mean', 0) for r in enhanced_results.values()]):.4f}")
    print(f"Vocabulary Size: {len(preprocessor.vectorizer.vocabulary_):,} features")
    print(f"Models saved to: {model_dir}/")
    
    return enhanced_results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Large-Scale Fake News Detector Training'
    )
    
    parser.add_argument('--data', type=str, 
                       default='data/large_scale_fake_news_dataset.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing (default: full dataset)')
    parser.add_argument('--max-features', type=int, default=20000,
                       help='Maximum TF-IDF features (default: 20000)')
    parser.add_argument('--model-dir', type=str, default='models_large_scale',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Run training
    results = train_large_scale(
        data_path=args.data,
        sample_size=args.sample,
        max_features=args.max_features,
        model_dir=args.model_dir
    )
    
    if results:
        print(f"\n✅ Training completed successfully!")
        # Print final comparison
        print("\n📊 Final Model Comparison:")
        for model_name, result in results.items():
            cv_score = result.get('cv_mean', 0)
            test_acc = result.get('test_accuracy', 0)
            print(f"   {model_name:20s}: CV={cv_score:.4f}, Test={test_acc:.4f}")
    else:
        print(f"\n❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()