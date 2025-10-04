import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from preprocess import NewsPreprocessor
from model import FakeNewsClassifier

class FakeNewsPredictor:
    """
    A comprehensive prediction system for fake news detection.
    Handles loading trained models and making predictions on new text data.
    """
    
    def __init__(self, model_path=None, vectorizer_path=None):
        self.preprocessor = NewsPreprocessor()
        self.classifier = FakeNewsClassifier()
        self.model_loaded = False
        self.vectorizer_loaded = False
        
        # Load models if paths provided
        if model_path:
            self.load_model(model_path)
        if vectorizer_path:
            self.load_vectorizer(vectorizer_path)
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            model = joblib.load(model_path)
            # Determine model type and load appropriately
            model_name = type(model).__name__.lower()
            if 'naive' in model_name:
                model_key = 'naive_bayes'
            elif 'svc' in model_name:
                model_key = 'svm'
            elif 'random' in model_name:
                model_key = 'random_forest'
            elif 'logistic' in model_name:
                model_key = 'logistic_regression'
            else:
                model_key = 'loaded_model'
            
            self.classifier.trained_models[model_key] = model
            self.classifier.best_model = model
            self.classifier.best_model_name = model_key
            self.model_loaded = True
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_vectorizer(self, vectorizer_path):
        """
        Load a trained vectorizer from disk.
        
        Args:
            vectorizer_path (str): Path to the saved vectorizer
        """
        try:
            self.preprocessor.load_vectorizer(vectorizer_path)
            self.vectorizer_loaded = True
            print(f"Vectorizer loaded successfully from {vectorizer_path}")
            
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            raise
    
    def predict_single(self, text, return_probability=False):
        """
        Predict whether a single text is fake news or not.
        
        Args:
            text (str): Text to analyze
            return_probability (bool): Whether to return prediction probabilities
            
        Returns:
            dict: Prediction results
        """
        if not self.model_loaded:
            raise ValueError("No model loaded. Please load a model first.")
        if not self.vectorizer_loaded:
            raise ValueError("No vectorizer loaded. Please load a vectorizer first.")
        
        # Preprocess the text
        processed_text = self.preprocessor.preprocess_text(text)
        
        # Transform to TF-IDF
        text_tfidf = self.preprocessor.transform_texts([processed_text])
        
        # Make prediction
        prediction = self.classifier.predict(text_tfidf)[0]
        
        # Get probability if requested
        probability = None
        confidence = None
        if return_probability:
            try:
                prob_array = self.classifier.predict_proba(text_tfidf)[0]
                probability = {
                    'real': float(prob_array[0]),
                    'fake': float(prob_array[1])
                }
                confidence = float(max(prob_array))
            except:
                print("Warning: Model does not support probability predictions")
        
        # Interpret prediction
        label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
        
        result = {
            'text': text,
            'processed_text': processed_text,
            'prediction': int(prediction),
            'label': label,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': self.classifier.best_model_name
        }
        
        if probability:
            result['probability'] = probability
            result['confidence'] = confidence
        
        return result
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        Predict multiple texts at once.
        
        Args:
            texts (list): List of texts to analyze
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            list: List of prediction results
        """
        if not self.model_loaded:
            raise ValueError("No model loaded. Please load a model first.")
        if not self.vectorizer_loaded:
            raise ValueError("No vectorizer loaded. Please load a vectorizer first.")
        
        results = []
        
        for text in texts:
            try:
                result = self.predict_single(text, return_probabilities)
                results.append(result)
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e),
                    'prediction': None,
                    'label': 'ERROR'
                })
        
        return results
    
    def predict_from_file(self, file_path, text_column='text', output_path=None):
        """
        Predict fake news from a CSV file.
        
        Args:
            file_path (str): Path to CSV file containing texts
            text_column (str): Name of column containing text data
            output_path (str): Path to save results (optional)
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in file")
        
        # Make predictions
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts, return_probabilities=True)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['prediction'] = [p['prediction'] for p in predictions]
        results_df['label'] = [p['label'] for p in predictions]
        results_df['confidence'] = [p.get('confidence', 0) for p in predictions]
        results_df['fake_probability'] = [p.get('probability', {}).get('fake', 0) for p in predictions]
        results_df['real_probability'] = [p.get('probability', {}).get('real', 0) for p in predictions]
        
        # Save results if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return results_df
    
    def get_prediction_summary(self, predictions):
        """
        Generate a summary of predictions.
        
        Args:
            predictions (list): List of prediction results
            
        Returns:
            dict: Summary statistics
        """
        if not predictions:
            return {}
        
        # Filter out errors
        valid_predictions = [p for p in predictions if 'error' not in p]
        
        if not valid_predictions:
            return {'error': 'No valid predictions found'}
        
        total_count = len(valid_predictions)
        fake_count = sum(1 for p in valid_predictions if p['prediction'] == 1)
        real_count = total_count - fake_count
        
        # Calculate confidence statistics if available
        confidences = [p.get('confidence') for p in valid_predictions if p.get('confidence')]
        
        summary = {
            'total_predictions': total_count,
            'fake_news_count': fake_count,
            'real_news_count': real_count,
            'fake_news_percentage': (fake_count / total_count) * 100 if total_count > 0 else 0,
            'real_news_percentage': (real_count / total_count) * 100 if total_count > 0 else 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if confidences:
            summary['avg_confidence'] = np.mean(confidences)
            summary['min_confidence'] = np.min(confidences)
            summary['max_confidence'] = np.max(confidences)
        
        return summary
    
    def interactive_prediction(self):
        """
        Interactive command-line prediction interface.
        """
        if not self.model_loaded or not self.vectorizer_loaded:
            print("Please load both model and vectorizer before using interactive mode.")
            return
        
        print("\n" + "="*60)
        print("     FAKE NEWS DETECTOR - Interactive Mode")
        print("="*60)
        print("Enter text to analyze (type 'quit' to exit):\n")
        
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
                result = self.predict_single(text, return_probability=True)
                
                # Display results
                print("\n" + "-"*40)
                print(f"PREDICTION: {result['label']}")
                
                if 'confidence' in result:
                    print(f"CONFIDENCE: {result['confidence']:.2%}")
                
                if 'probability' in result:
                    print(f"REAL NEWS: {result['probability']['real']:.2%}")
                    print(f"FAKE NEWS: {result['probability']['fake']:.2%}")
                
                print(f"MODEL USED: {result['model_used']}")
                print("-"*40 + "\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def save_predictions(self, predictions, output_path):
        """
        Save predictions to a JSON file.
        
        Args:
            predictions (list): List of prediction results
            output_path (str): Path to save the predictions
        """
        import json
        
        # Prepare data for JSON serialization
        json_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': self.classifier.best_model_name if self.model_loaded else 'unknown',
            'total_predictions': len(predictions),
            'predictions': predictions,
            'summary': self.get_prediction_summary(predictions)
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Predictions saved to {output_path}")

def quick_predict(text, model_path=None, vectorizer_path=None):
    """
    Quick prediction function for single text analysis.
    
    Args:
        text (str): Text to analyze
        model_path (str): Path to trained model (optional)
        vectorizer_path (str): Path to trained vectorizer (optional)
        
    Returns:
        dict: Prediction result
    """
    predictor = FakeNewsPredictor(model_path, vectorizer_path)
    
    # If no models provided, train on comprehensive dataset
    if not predictor.model_loaded:
        print("No model provided. Training on comprehensive dataset...")
        from preprocess import load_comprehensive_dataset
        from sklearn.model_selection import train_test_split
        
        # Load comprehensive dataset
        sample_df = load_comprehensive_dataset()
        texts, labels = predictor.preprocessor.prepare_data(sample_df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.3, random_state=42
        )
        
        predictor.preprocessor.fit_vectorizer(X_train)
        X_train_tfidf = predictor.preprocessor.transform_texts(X_train)
        
        predictor.classifier.train_single_model('naive_bayes', X_train_tfidf, y_train)
        
        # Set the trained model as the best model
        predictor.classifier.best_model = predictor.classifier.trained_models['naive_bayes']
        predictor.classifier.best_model_name = 'naive_bayes'
        
        predictor.model_loaded = True
        predictor.vectorizer_loaded = True
    
    return predictor.predict_single(text, return_probability=True)

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Fake News Predictor...")
    
    # Test with sample texts
    sample_texts = [
        "Scientists discover cure for cancer, but pharmaceutical companies don't want you to know!",
        "The Federal Reserve announced interest rate changes following economic indicators.",
        "SHOCKING: Celebrity scandal rocks Hollywood, career destroyed!",
        "Local university receives federal funding for climate research project."
    ]
    
    # Quick predictions using sample model
    print("\nMaking quick predictions...")
    for i, text in enumerate(sample_texts, 1):
        print(f"\n--- Text {i} ---")
        print(f"Text: {text}")
        
        result = quick_predict(text)
        print(f"Prediction: {result['label']}")
        
        if 'confidence' in result:
            print(f"Confidence: {result['confidence']:.2%}")
    
    # Interactive mode demo (commented out for automated testing)
    # predictor = FakeNewsPredictor()
    # predictor.interactive_prediction()