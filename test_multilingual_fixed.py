#!/usr/bin/env python3
"""
Multilingual Fake News Detection Testing Script
Tests the enhanced multilingual model with comprehensive language support
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from multilingual_preprocessing import MultilingualPreprocessor
import warnings
warnings.filterwarnings('ignore')

class MultilingualFakeNewsDetector:
    """
    Enhanced multilingual fake news detector with support for
    English, Spanish, French, and Hindi.
    """
    
    def __init__(self, models_dir="models_multilingual"):
        self.models_dir = Path(models_dir)
        self.preprocessor = None
        self.best_model = None
        
        # Load trained models
        self.load_models()
    
    def load_models(self):
        """Load the trained multilingual models."""
        print("🔄 Loading multilingual models...")
        
        # Load preprocessor
        preprocessor_file = self.models_dir / 'multilingual_preprocessor.pkl'
        if preprocessor_file.exists():
            with open(preprocessor_file, 'rb') as f:
                self.preprocessor = pickle.load(f)
            print("   ✅ Preprocessor loaded")
        else:
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_file}")
        
        # Load best model
        best_model_file = self.models_dir / 'multilingual_best_model.pkl'
        if best_model_file.exists():
            with open(best_model_file, 'rb') as f:
                self.best_model = pickle.load(f)
            print("   ✅ Best model loaded")
        else:
            raise FileNotFoundError(f"Best model not found: {best_model_file}")
        
        print("🌐 Multilingual fake news detector ready!")
        print("   📝 Supported languages: English, Spanish, French, Hindi")
    
    def predict_single(self, text, language=None):
        """Predict fake news for a single text."""
        # Create dataframe
        df = pd.DataFrame({
            'text': [text],
            'label': [0],  # Placeholder
            'language': [language] if language else [None]
        })
        
        # Preprocess (suppress output)
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            processed_df = self.preprocessor.preprocess_dataset(df)
            X = self.preprocessor.vectorize_texts(processed_df['processed_text'], fit=False)
        finally:
            sys.stdout = old_stdout
        
        # Predict
        prediction = self.best_model.predict(X)[0]
        confidence = self.best_model.predict_proba(X)[0].max()
        
        return {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': confidence,
            'detected_language': processed_df['language'].iloc[0] if 'language' in processed_df.columns else 'unknown'
        }
    
    def comprehensive_test(self):
        """Run comprehensive multilingual tests."""
        print("🧪" * 20)
        print("  COMPREHENSIVE MULTILINGUAL TESTING")
        print("🧪" * 20)
        
        # Test samples for different languages
        test_samples = {
            'English': [
                {
                    'text': "Scientists have discovered a breakthrough cure for cancer using only natural ingredients.",
                    'expected': 'Fake',
                    'type': 'Health Misinformation'
                },
                {
                    'text': "The Federal Reserve announced new monetary policy changes following extensive economic analysis.",
                    'expected': 'Real',
                    'type': 'Economic News'
                },
                {
                    'text': "BREAKING: This one weird trick will eliminate all your debt overnight - banks hate it!",
                    'expected': 'Fake',
                    'type': 'Financial Scam'
                },
                {
                    'text': "Researchers at MIT published findings on quantum computing advances in Nature journal.",
                    'expected': 'Real',
                    'type': 'Scientific News'
                }
            ],
            'Spanish': [
                {
                    'text': "URGENTE: Los médicos descubrieron esta hierba milagrosa que cura la diabetes en 24 horas.",
                    'expected': 'Fake',
                    'type': 'Fake Health News'
                },
                {
                    'text': "El Ministerio de Salud anunció nuevas directrices para la vacunación tras estudios científicos.",
                    'expected': 'Real',
                    'type': 'Official Health News'
                },
                {
                    'text': "EXCLUSIVO: La verdad que las farmacéuticas no quieren que sepas sobre este remedio casero.",
                    'expected': 'Fake',
                    'type': 'Conspiracy Theory'
                },
                {
                    'text': "Según el último informe del Instituto Nacional de Estadística, la economía muestra recuperación.",
                    'expected': 'Real',
                    'type': 'Economic Report'
                }
            ],
            'French': [
                {
                    'text': "RÉVÉLATION: Ce remède de grand-mère guérit toutes les maladies en une semaine seulement!",
                    'expected': 'Fake',
                    'type': 'Health Misinformation'
                },
                {
                    'text': "Le ministère de la Santé a publié de nouvelles recommandations basées sur des études cliniques.",
                    'expected': 'Real',
                    'type': 'Official Health News'
                },
                {
                    'text': "URGENT: Les scientifiques cachent cette découverte qui pourrait révolutionner la médecine.",
                    'expected': 'Fake',
                    'type': 'Science Conspiracy'
                },
                {
                    'text': "L'université de Sorbonne a publié des recherches sur l'intelligence artificielle dans Science.",
                    'expected': 'Real',
                    'type': 'Academic News'
                }
            ],
            'Hindi': [
                {
                    'text': "एक्सक्लूसिव: यह घरेलू नुस्खा 7 दिन में सभी बीमारियों को ठीक कर देता है!",
                    'expected': 'Fake',
                    'type': 'Health Misinformation'
                },
                {
                    'text': "स्वास्थ्य मंत्रालय ने नैदानिक अध्ययन के आधार पर नई दिशानिर्देश जारी किए।",
                    'expected': 'Real',
                    'type': 'Official Health News'
                },
                {
                    'text': "तत्काल: डॉक्टर इस चमत्कारी उपचार को छुपाते हैं जो सभी रोगों का इलाज करता है।",
                    'expected': 'Fake',
                    'type': 'Medical Conspiracy'
                },
                {
                    'text': "दिल्ली विश्वविद्यालय के शोधकर्ताओं ने आयुर्वेद पर अनुसंधान प्रकाशित किया।",
                    'expected': 'Real',
                    'type': 'Academic Research'
                }
            ]
        }
        
        total_correct = 0
        total_tests = 0
        language_results = {}
        
        for language, samples in test_samples.items():
            print(f"\n🌍 Testing {language} samples...")
            
            correct = 0
            confidence_scores = []
            
            for i, sample in enumerate(samples, 1):
                result = self.predict_single(sample['text'], language.lower())
                
                is_correct = result['prediction'] == sample['expected']
                if is_correct:
                    correct += 1
                
                confidence_scores.append(result['confidence'])
                
                status = "✅" if is_correct else "❌"
                print(f"   {status} Test {i}: {sample['type']}")
                print(f"      Expected: {sample['expected']}, Got: {result['prediction']} (Confidence: {result['confidence']:.3f})")
                print(f"      Detected Language: {result['detected_language']}")
                print(f"      Text: {sample['text'][:80]}...")
            
            accuracy = correct / len(samples)
            avg_confidence = np.mean(confidence_scores)
            
            language_results[language] = {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'correct': correct,
                'total': len(samples)
            }
            
            total_correct += correct
            total_tests += len(samples)
            
            print(f"\n   📊 {language} Results:")
            print(f"      Accuracy: {accuracy:.3f} ({correct}/{len(samples)})")
            print(f"      Average Confidence: {avg_confidence:.3f}")
        
        # Overall results
        overall_accuracy = total_correct / total_tests
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   📊 Total Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_tests})")
        print(f"   🌍 Languages Tested: {len(test_samples)}")
        
        # Language-wise summary
        print(f"\n📈 Language-wise Performance Summary:")
        for lang, results in language_results.items():
            print(f"   {lang}: {results['accuracy']:.3f} accuracy, {results['avg_confidence']:.3f} confidence")
        
        # Identify any concerning patterns
        low_confidence_langs = [lang for lang, res in language_results.items() if res['avg_confidence'] < 0.8]
        if low_confidence_langs:
            print(f"\n⚠️  Languages with lower confidence (< 0.8): {', '.join(low_confidence_langs)}")
        else:
            print(f"\n✅ All languages show high confidence (>= 0.8)!")
        
        return language_results
    
    def interactive_test(self):
        """Interactive testing mode for user input."""
        print(f"\n🤖 Interactive Multilingual Fake News Detector")
        print(f"   📝 Supported languages: English, Spanish, French, Hindi")
        print(f"   💡 Type 'quit' to exit")
        
        while True:
            print(f"\n" + "="*50)
            user_input = input("Enter news text to analyze: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                print("⚠️  Please enter some text.")
                continue
            
            try:
                result = self.predict_single(user_input)
                
                print(f"\n🔍 Analysis Results:")
                print(f"   📰 Prediction: {result['prediction']}")
                print(f"   📊 Confidence: {result['confidence']:.3f}")
                print(f"   🌍 Detected Language: {result['detected_language']}")
                
                # Confidence interpretation
                if result['confidence'] >= 0.9:
                    conf_level = "Very High"
                elif result['confidence'] >= 0.8:
                    conf_level = "High"
                elif result['confidence'] >= 0.7:
                    conf_level = "Moderate"
                else:
                    conf_level = "Low"
                
                print(f"   💪 Confidence Level: {conf_level}")
                
            except Exception as e:
                print(f"❌ Error during analysis: {str(e)}")

def main():
    """Main execution function."""
    try:
        # Initialize detector
        detector = MultilingualFakeNewsDetector()
        
        # Run comprehensive test
        detector.comprehensive_test()
        
        # Ask user if they want interactive mode
        while True:
            choice = input("\nWould you like to try interactive testing? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                detector.interactive_test()
                break
            elif choice in ['n', 'no']:
                print("Testing complete! 🎉")
                break
            else:
                print("Please enter 'y' or 'n'")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()