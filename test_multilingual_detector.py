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
        
        # Preprocess without printing (quiet mode)
        processed_df = self.preprocessor.preprocess_dataset(df)
        
        # Vectorize
        X = self.preprocessor.vectorize_texts(processed_df['processed_text'], fit=False)
        
        # Predict
        prediction = self.best_model.predict(X)[0]
        confidence = self.best_model.predict_proba(X)[0].max()
        
        return {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': confidence,
            'detected_language': processed_df['language'].iloc[0] if 'language' in processed_df.columns else 'unknown'
        }
    
    def batch_predict(self, texts, languages=None):
        """Predict fake news for multiple texts."""
        if languages is None:
            languages = [None] * len(texts)
        
        results = []
        for i, (text, lang) in enumerate(zip(texts, languages)):
            result = self.predict_single(text, lang)
            result['text'] = text[:100] + "..." if len(text) > 100 else text
            results.append(result)
        
        return results
    
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
        }\n        \n        total_correct = 0\n        total_tests = 0\n        language_results = {}\n        \n        for language, samples in test_samples.items():\n            print(f\"\\n🌍 Testing {language} samples...\")\n            \n            correct = 0\n            confidence_scores = []\n            \n            for i, sample in enumerate(samples, 1):\n                result = self.predict_single(sample['text'], language.lower())\n                \n                is_correct = result['prediction'] == sample['expected']\n                if is_correct:\n                    correct += 1\n                \n                confidence_scores.append(result['confidence'])\n                \n                status = \"✅\" if is_correct else \"❌\"\n                print(f\"   {status} Test {i}: {sample['type']}\")\n                print(f\"      Expected: {sample['expected']}, Got: {result['prediction']} (Confidence: {result['confidence']:.3f})\")\n                print(f\"      Detected Language: {result['detected_language']}\")\n                print(f\"      Text: {sample['text'][:80]}...\")\n            \n            accuracy = correct / len(samples)\n            avg_confidence = np.mean(confidence_scores)\n            \n            language_results[language] = {\n                'accuracy': accuracy,\n                'avg_confidence': avg_confidence,\n                'correct': correct,\n                'total': len(samples)\n            }\n            \n            total_correct += correct\n            total_tests += len(samples)\n            \n            print(f\"\\n   📊 {language} Results:\")\n            print(f\"      Accuracy: {accuracy:.3f} ({correct}/{len(samples)})\")\n            print(f\"      Average Confidence: {avg_confidence:.3f}\")\n        \n        # Overall results\n        overall_accuracy = total_correct / total_tests\n        print(f\"\\n🎯 OVERALL RESULTS:\")\n        print(f\"   📊 Total Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_tests})\")\n        print(f\"   🌍 Languages Tested: {len(test_samples)}\")\n        \n        # Language-wise summary\n        print(f\"\\n📈 Language-wise Performance Summary:\")\n        for lang, results in language_results.items():\n            print(f\"   {lang}: {results['accuracy']:.3f} accuracy, {results['avg_confidence']:.3f} confidence\")\n        \n        # Identify any concerning patterns\n        low_confidence_langs = [lang for lang, res in language_results.items() if res['avg_confidence'] < 0.8]\n        if low_confidence_langs:\n            print(f\"\\n⚠️  Languages with lower confidence (< 0.8): {', '.join(low_confidence_langs)}\")\n        else:\n            print(f\"\\n✅ All languages show high confidence (>= 0.8)!\")\n        \n        return language_results\n    \n    def interactive_test(self):\n        \"\"\"Interactive testing mode for user input.\"\"\"\n        print(f\"\\n🤖 Interactive Multilingual Fake News Detector\")\n        print(f\"   📝 Supported languages: English, Spanish, French, Hindi\")\n        print(f\"   💡 Type 'quit' to exit\")\n        \n        while True:\n            print(f\"\\n\" + \"=\"*50)\n            user_input = input(\"Enter news text to analyze: \").strip()\n            \n            if user_input.lower() in ['quit', 'exit', 'q']:\n                print(\"👋 Goodbye!\")\n                break\n            \n            if not user_input:\n                print(\"⚠️  Please enter some text.\")\n                continue\n            \n            try:\n                result = self.predict_single(user_input)\n                \n                print(f\"\\n🔍 Analysis Results:\")\n                print(f\"   📰 Prediction: {result['prediction']}\")\n                print(f\"   📊 Confidence: {result['confidence']:.3f}\")\n                print(f\"   🌍 Detected Language: {result['detected_language']}\")\n                \n                # Confidence interpretation\n                if result['confidence'] >= 0.9:\n                    conf_level = \"Very High\"\n                elif result['confidence'] >= 0.8:\n                    conf_level = \"High\"\n                elif result['confidence'] >= 0.7:\n                    conf_level = \"Moderate\"\n                else:\n                    conf_level = \"Low\"\n                \n                print(f\"   💪 Confidence Level: {conf_level}\")\n                \n            except Exception as e:\n                print(f\"❌ Error during analysis: {str(e)}\")\n\ndef main():\n    \"\"\"Main execution function.\"\"\"\n    try:\n        # Initialize detector\n        detector = MultilingualFakeNewsDetector()\n        \n        # Run comprehensive test\n        detector.comprehensive_test()\n        \n        # Ask user if they want interactive mode\n        while True:\n            choice = input(\"\\nWould you like to try interactive testing? (y/n): \").strip().lower()\n            if choice in ['y', 'yes']:\n                detector.interactive_test()\n                break\n            elif choice in ['n', 'no']:\n                print(\"Testing complete! 🎉\")\n                break\n            else:\n                print(\"Please enter 'y' or 'n'\")\n        \n    except Exception as e:\n        print(f\"❌ Error: {str(e)}\")\n        import traceback\n        traceback.print_exc()\n\nif __name__ == \"__main__\":\n    main()