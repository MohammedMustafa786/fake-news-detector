#!/usr/bin/env python3
"""
Fake News Detector - Main Application

A comprehensive fake news detection system using machine learning.
This script demonstrates the complete workflow from data preprocessing
to model training and prediction.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import NewsPreprocessor, create_sample_data, load_comprehensive_dataset
from src.model import FakeNewsClassifier
from src.predict import FakeNewsPredictor, quick_predict

def train_model(data_path=None, save_model=True, model_dir='models'):
    """
    Train the fake news detection model.
    
    Args:
        data_path (str): Path to training data CSV file
        save_model (bool): Whether to save the trained model
        model_dir (str): Directory to save models
    
    Returns:
        tuple: (classifier, preprocessor, results)
    """
    print("=" * 60)
    print("FAKE NEWS DETECTOR - TRAINING MODE")
    print("=" * 60)
    
    # Load or create data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("Using comprehensive dataset...")
        if data_path:
            print(f"Warning: File {data_path} not found, using comprehensive dataset")
        df = load_comprehensive_dataset()
    
    print(f"Dataset size: {len(df)} articles")
    print(f"Fake news: {sum(df['label'])} articles")
    print(f"Real news: {len(df) - sum(df['label'])} articles")
    
    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = NewsPreprocessor(max_features=5000)
    
    # Prepare data
    texts, labels = preprocessor.prepare_data(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Fit vectorizer and transform data
    preprocessor.fit_vectorizer(X_train)
    X_train_tfidf = preprocessor.transform_texts(X_train)
    X_test_tfidf = preprocessor.transform_texts(X_test)
    
    # Train classifier
    print("\nTraining models...")
    classifier = FakeNewsClassifier()
    results = classifier.train_all_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Display results
    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    
    comparison_df = classifier.get_model_comparison()
    print(comparison_df.to_string(index=False))
    
    print(f"\nBest Model: {classifier.best_model_name}")
    
    # Save models if requested
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        
        # Save best model
        best_model_path = os.path.join(model_dir, f'best_model_{classifier.best_model_name}.pkl')
        classifier.save_model(classifier.best_model_name, best_model_path)
        
        # Save vectorizer
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        preprocessor.save_vectorizer(vectorizer_path)
        
        # Save all models
        for model_name in classifier.trained_models.keys():
            model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
            classifier.save_model(model_name, model_path)
        
        # Save results
        results_path = os.path.join(model_dir, 'training_results.csv')
        comparison_df.to_csv(results_path, index=False)
        
        print(f"\nModels saved to {model_dir}/")
    
    return classifier, preprocessor, results

def predict_text(text, model_dir='models'):
    """
    Predict whether a single text is fake news.
    
    Args:
        text (str): Text to analyze
        model_dir (str): Directory containing saved models
        
    Returns:
        dict: Prediction results
    """
    # Try to load saved models
    best_model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
    
    if best_model_files:
        model_path = os.path.join(model_dir, best_model_files[0])
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        
        if os.path.exists(vectorizer_path):
            predictor = FakeNewsPredictor(model_path, vectorizer_path)
            return predictor.predict_single(text, return_probability=True)
    
    # Fallback to quick predict with sample model
    print("No saved models found. Using sample model...")
    return quick_predict(text)

def predict_batch(texts, model_dir='models'):
    """
    Predict multiple texts.
    
    Args:
        texts (list): List of texts to analyze
        model_dir (str): Directory containing saved models
        
    Returns:
        list: Prediction results
    """
    # Try to load saved models
    best_model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
    
    if best_model_files:
        model_path = os.path.join(model_dir, best_model_files[0])
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        
        if os.path.exists(vectorizer_path):
            predictor = FakeNewsPredictor(model_path, vectorizer_path)
            return predictor.predict_batch(texts, return_probabilities=True)
    
    # Fallback to quick predict
    print("No saved models found. Using sample model...")
    results = []
    for text in texts:
        results.append(quick_predict(text))
    return results

def predict_from_file(input_path, output_path=None, model_dir='models'):
    """
    Predict fake news from a CSV file.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save results
        model_dir (str): Directory containing saved models
        
    Returns:
        pd.DataFrame: Results dataframe
    """
    # Try to load saved models
    best_model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
    
    if best_model_files:
        model_path = os.path.join(model_dir, best_model_files[0])
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        
        if os.path.exists(vectorizer_path):
            predictor = FakeNewsPredictor(model_path, vectorizer_path)
            return predictor.predict_from_file(input_path, output_path=output_path)
    
    # Fallback implementation
    print("No saved models found. Using sample model...")
    df = pd.read_csv(input_path)
    texts = df['text'].tolist()
    predictions = predict_batch(texts, model_dir)
    
    # Add predictions to dataframe
    df['prediction'] = [p['prediction'] for p in predictions]
    df['label'] = [p['label'] for p in predictions]
    df['confidence'] = [p.get('confidence', 0) for p in predictions]
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    return df

def interactive_mode(model_dir='models'):
    """
    Interactive prediction mode.
    
    Args:
        model_dir (str): Directory containing saved models
    """
    print("=" * 60)
    print("FAKE NEWS DETECTOR - INTERACTIVE MODE")
    print("=" * 60)
    print("Enter text to analyze (type 'quit' to exit)\\n")
    
    while True:
        try:
            text = input("Enter news text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            # Make prediction
            result = predict_text(text, model_dir)
            
            # Display results
            print("\\n" + "-"*40)
            print(f"PREDICTION: {result['label']}")
            
            if 'confidence' in result:
                print(f"CONFIDENCE: {result['confidence']:.2%}")
            
            if 'probability' in result:
                print(f"REAL NEWS: {result['probability']['real']:.2%}")
                print(f"FAKE NEWS: {result['probability']['fake']:.2%}")
            
            print(f"MODEL USED: {result.get('model_used', 'sample_model')}")
            print("-"*40 + "\\n")
            
        except KeyboardInterrupt:
            print("\\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def demo_mode():
    """
    Demonstration mode showing complete workflow.
    """
    print("=" * 60)
    print("FAKE NEWS DETECTOR - DEMO MODE")
    print("=" * 60)
    
    # Sample texts for demo
    demo_texts = [
        "BREAKING: Scientists discover aliens living among us, government tries to cover up!",
        "The Federal Reserve announced a 0.25% interest rate increase following today's meeting.",
        "You won't believe this miracle cure that doctors don't want you to know about!",
        "Climate scientists report record-breaking temperatures recorded this summer.",
        "SHOCKING: Celebrity caught in massive scandal, career over!",
        "Local university receives grant funding for renewable energy research project.",
    ]
    
    print("\\nDemonstrating fake news detection on sample texts...\\n")
    
    for i, text in enumerate(demo_texts, 1):
        print(f"--- Text {i} ---")
        print(f"Text: {text}")
        
        result = quick_predict(text)
        print(f"Prediction: {result['label']}")
        
        if 'confidence' in result:
            print(f"Confidence: {result['confidence']:.2%}")
        
        print("-" * 50)

def main():
    """
    Main application entry point.
    """
    parser = argparse.ArgumentParser(
        description='Fake News Detector - ML-based text classification system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                           # Train model with sample data
  python main.py --train --data data/news.csv     # Train with custom data
  python main.py --predict "Text to analyze"      # Predict single text
  python main.py --interactive                    # Interactive mode
  python main.py --demo                           # Run demonstration
  python main.py --batch data/test.csv results.csv # Batch prediction
        """
    )
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Train the model')
    group.add_argument('--predict', type=str, help='Predict single text')
    group.add_argument('--interactive', action='store_true', help='Interactive prediction mode')
    group.add_argument('--batch', nargs=2, metavar=('INPUT', 'OUTPUT'), 
                      help='Batch prediction from CSV file')
    group.add_argument('--demo', action='store_true', help='Run demonstration')
    
    # Optional arguments
    parser.add_argument('--data', type=str, help='Path to training data CSV file')
    parser.add_argument('--models', type=str, default='models', 
                       help='Directory for saving/loading models (default: models)')
    parser.add_argument('--no-save', action='store_true', 
                       help="Don't save trained models")
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    if not os.path.exists(args.models):
        os.makedirs(args.models)
    
    # Execute based on mode
    if args.train:
        train_model(
            data_path=args.data, 
            save_model=not args.no_save, 
            model_dir=args.models
        )
    
    elif args.predict:
        result = predict_text(args.predict, args.models)
        
        print("\\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['label']}")
        
        if 'confidence' in result:
            print(f"Confidence: {result['confidence']:.2%}")
        
        if 'probability' in result:
            print(f"Real News: {result['probability']['real']:.2%}")
            print(f"Fake News: {result['probability']['fake']:.2%}")
    
    elif args.interactive:
        interactive_mode(args.models)
    
    elif args.batch:
        input_path, output_path = args.batch
        results_df = predict_from_file(input_path, output_path, args.models)
        
        print("\\n" + "="*50)
        print("BATCH PREDICTION RESULTS")
        print("="*50)
        print(f"Total predictions: {len(results_df)}")
        print(f"Fake news detected: {sum(results_df['prediction'])} articles")
        print(f"Real news detected: {len(results_df) - sum(results_df['prediction'])} articles")
    
    elif args.demo:
        demo_mode()

if __name__ == "__main__":
    main()