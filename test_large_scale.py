#!/usr/bin/env python3
"""
Test script for the large-scale fake news detector.
Uses the trained 50k+ article models for high-accuracy detection.
"""
import os
import sys
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.large_scale_preprocess import LargeScaleNewsPreprocessor
from src.model import FakeNewsClassifier
import joblib

class LargeScaleFakeNewsPredictor:
    """
    High-performance fake news predictor using large-scale trained models.
    """
    
    def __init__(self, model_dir='models_large_scale'):
        self.model_dir = model_dir
        self.preprocessor = LargeScaleNewsPreprocessor()
        self.classifier = FakeNewsClassifier()
        self.model_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load the trained large-scale models."""
        try:
            # Load vectorizer
            vectorizer_path = os.path.join(self.model_dir, 'large_scale_vectorizer.pkl')
            if os.path.exists(vectorizer_path):
                self.preprocessor.load_vectorizer(vectorizer_path)
                print(f"✅ Loaded vectorizer from {vectorizer_path}")
            else:
                print(f"❌ Vectorizer not found: {vectorizer_path}")
                return False
            
            # Find and load best model
            best_model_files = [f for f in os.listdir(self.model_dir) 
                              if f.startswith('best_model_large_scale_') and f.endswith('.pkl')]
            
            if best_model_files:
                best_model_path = os.path.join(self.model_dir, best_model_files[0])
                model_name = best_model_files[0].replace('best_model_large_scale_', '').replace('.pkl', '')
                
                # Load the model
                model = joblib.load(best_model_path)
                self.classifier.trained_models[model_name] = model
                self.classifier.best_model = model
                self.classifier.best_model_name = model_name
                
                print(f"✅ Loaded best model: {model_name} from {best_model_path}")
                self.model_loaded = True
                return True
            else:
                print(f"❌ No best model found in {self.model_dir}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_single(self, text):
        """
        Predict if a single text is fake news.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Prediction results with confidence and probabilities
        """
        if not self.model_loaded:
            return {"error": "Models not loaded"}
        
        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess_text_parallel(text)
            
            # Transform to TF-IDF
            text_tfidf = self.preprocessor.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.classifier.best_model.predict(text_tfidf)[0]
            
            # Get probabilities if available
            try:
                probabilities = self.classifier.best_model.predict_proba(text_tfidf)[0]
                real_prob = probabilities[0]
                fake_prob = probabilities[1]
                confidence = max(real_prob, fake_prob)
            except:
                real_prob = fake_prob = 0.5
                confidence = 0.5
            
            # Determine label
            label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
            
            return {
                "text": text,
                "prediction": prediction,
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    "real": real_prob,
                    "fake": fake_prob
                },
                "model_used": self.classifier.best_model_name,
                "vocabulary_size": len(self.preprocessor.vectorizer.vocabulary_)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

def test_sample_texts():
    """Test with various sample texts."""
    
    print("🚀" * 15)
    print("  LARGE-SCALE FAKE NEWS DETECTOR TEST")
    print("🚀" * 15)
    
    # Initialize predictor
    predictor = LargeScaleFakeNewsPredictor()
    
    if not predictor.model_loaded:
        print("\n❌ Could not load models. Please train the models first by running:")
        print("   python train_large_scale_fixed.py --sample 5000")
        return
    
    print(f"\n📊 Model Info:")
    print(f"   Best Model: {predictor.classifier.best_model_name}")
    print(f"   Vocabulary Size: {len(predictor.preprocessor.vectorizer.vocabulary_):,} features")
    
    # Sample texts for testing
    test_texts = [
        # Fake news examples
        {
            "text": "BREAKING: Scientists discover aliens living among us, government tries to cover up the truth!",
            "expected": "FAKE"
        },
        {
            "text": "You won't believe this miracle cure that doctors don't want you to know about - it works in just 3 days!",
            "expected": "FAKE"
        },
        {
            "text": "SHOCKING: New study reveals vaccines contain microchips for government surveillance!",
            "expected": "FAKE"
        },
        {
            "text": "URGENT: 5G towers are secretly controlling your thoughts and behavior patterns!",
            "expected": "FAKE"
        },
        
        # Real news examples
        {
            "text": "The Federal Reserve announced a 0.25% interest rate increase following today's meeting.",
            "expected": "REAL"
        },
        {
            "text": "Researchers at Harvard University published peer-reviewed findings on climate change in Nature journal.",
            "expected": "REAL"
        },
        {
            "text": "The Department of Health issued new guidelines for flu vaccination based on CDC recommendations.",
            "expected": "REAL"
        },
        {
            "text": "Stock markets showed mixed results today with technology sector gaining 2.3% amid earnings reports.",
            "expected": "REAL"
        }
    ]
    
    print(f"\n🔬 Testing on {len(test_texts)} sample articles...")
    print("=" * 80)
    
    correct_predictions = 0
    
    for i, test_case in enumerate(test_texts, 1):
        text = test_case["text"]
        expected = test_case["expected"]
        
        print(f"\n📝 Test {i}/{len(test_texts)}")
        print(f"Text: {text}")
        print(f"Expected: {expected} NEWS")
        
        # Make prediction
        result = predictor.predict_single(text)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        # Display results
        print(f"Prediction: {result['label']} ({result['confidence']:.1%} confidence)")
        print(f"Probabilities: Real={result['probabilities']['real']:.1%}, Fake={result['probabilities']['fake']:.1%}")
        
        # Check if prediction is correct
        predicted_type = "FAKE" if result['prediction'] == 1 else "REAL"
        is_correct = predicted_type == expected
        
        if is_correct:
            print("✅ CORRECT!")
            correct_predictions += 1
        else:
            print("❌ INCORRECT!")
        
        print("-" * 60)
    
    # Final results
    accuracy = correct_predictions / len(test_texts)
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Correct Predictions: {correct_predictions}/{len(test_texts)}")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Model Used: {predictor.classifier.best_model_name}")
    
    if accuracy >= 0.8:
        print("🎉 Excellent performance!")
    elif accuracy >= 0.6:
        print("👍 Good performance!")
    else:
        print("⚠️ Performance could be improved")

def interactive_test():
    """Interactive testing mode."""
    
    print("\n" + "🔍" * 15)
    print("  INTERACTIVE TESTING MODE")
    print("🔍" * 15)
    print("Enter news text to analyze (type 'quit' to exit)")
    
    predictor = LargeScaleFakeNewsPredictor()
    
    if not predictor.model_loaded:
        print("\n❌ Could not load models. Please train first.")
        return
    
    while True:
        try:
            text = input("\n📰 Enter news text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            # Make prediction
            result = predictor.predict_single(text)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Display results
            print("\n" + "="*50)
            print(f"🤖 PREDICTION: {result['label']}")
            print(f"📊 CONFIDENCE: {result['confidence']:.1%}")
            print(f"📈 PROBABILITIES:")
            print(f"   Real News: {result['probabilities']['real']:.1%}")
            print(f"   Fake News: {result['probabilities']['fake']:.1%}")
            print(f"🎯 MODEL: {result['model_used']}")
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Large-Scale Fake News Detector')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--text', type=str, 
                       help='Single text to analyze')
    
    args = parser.parse_args()
    
    if args.text:
        # Single text prediction
        predictor = LargeScaleFakeNewsPredictor()
        if predictor.model_loaded:
            result = predictor.predict_single(args.text)
            if "error" not in result:
                print(f"\nText: {args.text}")
                print(f"Prediction: {result['label']} ({result['confidence']:.1%} confidence)")
            else:
                print(f"Error: {result['error']}")
        
    elif args.interactive:
        # Interactive mode
        interactive_test()
    else:
        # Default: run sample tests
        test_sample_texts()
        
        # Ask if user wants interactive mode
        choice = input("\n🤔 Would you like to try interactive mode? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_test()

if __name__ == "__main__":
    main()