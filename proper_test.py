#!/usr/bin/env python3

import pickle
import sys
import os

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_multilingual_detector():
    """Load the multilingual fake news detector."""
    try:
        with open('models_multilingual/multilingual_preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('models_multilingual/multilingual_best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Models loaded successfully")
        return preprocessor, model
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None

def predict_news_article(text, preprocessor, model):
    """Predict if a news article is fake or real."""
    try:
        # Use the preprocessor's preprocess method
        processed_text = preprocessor.preprocess(text)
        processed_features = preprocessor.vectorizer.transform([processed_text])
        
        prediction = model.predict(processed_features)[0]
        probabilities = model.predict_proba(processed_features)[0]
        confidence = max(probabilities)
        
        return prediction, confidence, probabilities
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None, None

def main():
    print("🧪 PROPER NEWS ARTICLE TESTING")
    print("="*50)
    
    # Load models
    preprocessor, model = load_multilingual_detector()
    if preprocessor is None or model is None:
        print("Failed to load models")
        return
    
    # Test cases with proper news article format
    test_cases = [
        {
            "title": "FAKE Sports News (Rohit Sharma football)",
            "text": """
BREAKING NEWS: Rohit Sharma Announces Shocking Career Change to Professional Football
Mumbai, India - In an unprecedented move that has stunned the cricket world, Indian cricket captain Rohit Sharma announced today that he is leaving cricket to join Manchester United as a professional footballer. The 36-year-old opening batsman revealed he has been secretly training with football coaches for two years. Cricket experts are calling this the most shocking career change in sports history. Sharma claims he has always dreamed of playing football and believes his athletic skills will transfer to the pitch.
            """,
            "expected": "Fake"
        },
        {
            "title": "REAL Sports News (Rohit Sharma cricket)",
            "text": """
Indian Cricket Captain Rohit Sharma Leads Team to Victory Against Australia
Mumbai - Indian cricket team captain Rohit Sharma scored a magnificent 150 runs in the third Test match against Australia at the MCG, leading India to a comprehensive victory. The 36-year-old right-handed opener displayed exceptional batting skills throughout the innings, hitting 18 boundaries and 3 sixes. This victory puts India ahead 2-1 in the four-match Test series. Team management and fans have praised Sharma's leadership and batting performance.
            """,
            "expected": "Real"
        },
        {
            "title": "FAKE Science News (Moon on Earth)",
            "text": """
SHOCKING DISCOVERY: NASA Confirms Moon Has Been on Earth's Surface All Along
Washington D.C. - In a groundbreaking announcement that challenges fundamental astronomy, NASA scientists revealed today that the moon has actually been located on Earth's surface throughout history. Dr. Jonathan Newton, lead researcher at NASA's Goddard Space Flight Center, claims that all previous observations of the moon in space were optical illusions caused by atmospheric refraction. This discovery allegedly explains why lunar missions were so easy to accomplish and why we can see the moon so clearly from Earth.
            """,
            "expected": "Fake"
        },
        {
            "title": "REAL Science News (Moon research)",
            "text": """
NASA Announces New Lunar Research Program for 2024
Washington - The National Aeronautics and Space Administration has announced a comprehensive lunar research program scheduled to begin in 2024. The program will focus on studying moon rock samples collected during previous Apollo missions and analyze lunar surface composition. Scientists from multiple universities will participate in this research initiative. The program aims to better understand lunar formation and its relationship to Earth's geological history.
            """,
            "expected": "Real"
        }
    ]
    
    print(f"Testing {len(test_cases)} news articles:")
    print()
    
    correct_predictions = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🧪 Test {i}: {test_case['title']}")
        
        prediction, confidence, probabilities = predict_news_article(
            test_case['text'], preprocessor, model
        )
        
        if prediction is not None:
            # Convert prediction to human-readable format
            pred_label = "Fake" if prediction == 1 else "Real"
            
            # Check if correct
            is_correct = pred_label == test_case['expected']
            correct_predictions += is_correct
            
            status_emoji = "✅" if is_correct else "❌"
            
            print(f"   {status_emoji} Expected: {test_case['expected']}, Got: {pred_label}")
            print(f"   📊 Confidence: {confidence:.1%}")
            print(f"   📈 Probabilities: Real {probabilities[0]:.1%}, Fake {probabilities[1]:.1%}")
        else:
            print("   ❌ Prediction failed")
        
        print()
    
    accuracy = correct_predictions / len(test_cases)
    print("="*50)
    print(f"📊 FINAL RESULTS:")
    print(f"   Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_cases)})")
    
    if accuracy >= 0.75:
        print("   🎉 Good performance! The model works well with proper news articles.")
    else:
        print("   ⚠️  Model needs improvement for these types of articles.")

if __name__ == "__main__":
    main()