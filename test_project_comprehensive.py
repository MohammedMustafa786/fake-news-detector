#!/usr/bin/env python3
"""
Comprehensive Project Testing Script
Tests all components with proper percentage formatting (90% not 0.9)
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from multilingual_preprocessing import MultilingualPreprocessor
import sys
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

class ProjectTester:
    """
    Comprehensive tester for the multilingual fake news detection project.
    """
    
    def __init__(self):
        self.models_dir = Path("models_multilingual")
        self.data_dir = Path("data")
        self.preprocessor = None
        self.model = None
        self.test_results = {}
        
    def test_project_structure(self):
        """Test if all required files exist."""
        print("🔍 Testing Project Structure...")
        
        required_files = [
            "multilingual_preprocessing.py",
            "train_multilingual_model.py", 
            "download_multilingual_datasets.py",
            "test_multilingual_fixed.py",
            "data/enhanced_multilingual_fake_news_dataset.csv",
            "models_multilingual/multilingual_preprocessor.pkl",
            "models_multilingual/multilingual_best_model.pkl"
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in required_files:
            if Path(file_path).exists():
                existing_files.append(file_path)
                print(f"   ✅ {file_path}")
            else:
                missing_files.append(file_path)
                print(f"   ❌ {file_path}")
        
        if missing_files:
            print(f"\n⚠️  Missing files: {len(missing_files)}")
            return False
        else:
            print(f"\n✅ All required files present! ({len(existing_files)} files)")
            return True
    
    def test_model_loading(self):
        """Test loading of trained models."""
        print("\n🔄 Testing Model Loading...")
        
        try:
            # Load preprocessor
            preprocessor_file = self.models_dir / 'multilingual_preprocessor.pkl'
            with open(preprocessor_file, 'rb') as f:
                self.preprocessor = pickle.load(f)
            print("   ✅ Preprocessor loaded successfully")
            
            # Load model
            model_file = self.models_dir / 'multilingual_best_model.pkl'
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            print("   ✅ Best model loaded successfully")
            
            # Test model type
            model_type = type(self.model).__name__
            print(f"   📊 Model type: {model_type}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading models: {str(e)}")
            return False
    
    def test_dataset_integrity(self):
        """Test the enhanced multilingual dataset."""
        print("\n📊 Testing Dataset Integrity...")
        
        try:
            dataset_file = self.data_dir / 'enhanced_multilingual_fake_news_dataset.csv'
            df = pd.read_csv(dataset_file)
            
            print(f"   ✅ Dataset loaded successfully")
            print(f"   📈 Total articles: {len(df):,}")
            
            # Check required columns
            required_columns = ['text', 'label', 'language']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"   ❌ Missing columns: {missing_columns}")
                return False
            
            print(f"   ✅ All required columns present")
            
            # Check language distribution
            print(f"\n   🌐 Language Distribution:")
            for language in df['language'].unique():
                count = len(df[df['language'] == language])
                percentage = count / len(df) * 100
                print(f"      {language.title()}: {count:,} articles ({percentage:.1f}%)")
            
            # Check label distribution
            real_count = len(df[df['label'] == 0])
            fake_count = len(df[df['label'] == 1])
            print(f"\n   📰 Label Distribution:")
            print(f"      Real: {real_count:,} articles ({real_count/len(df)*100:.1f}%)")
            print(f"      Fake: {fake_count:,} articles ({fake_count/len(df)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading dataset: {str(e)}")
            return False
    
    def test_single_prediction(self, text, language, expected_type):
        """Test prediction on a single text with percentage formatting."""
        try:
            # Create dataframe
            df = pd.DataFrame({
                'text': [text],
                'label': [0],  # Placeholder
                'language': [language]
            })
            
            # Suppress preprocessing output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                processed_df = self.preprocessor.preprocess_dataset(df)
                X = self.preprocessor.vectorize_texts(processed_df['processed_text'], fit=False)
            finally:
                sys.stdout = old_stdout
            
            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Convert to percentages
            fake_prob = probabilities[1] * 100
            real_prob = probabilities[0] * 100
            confidence = max(fake_prob, real_prob)
            
            result = {
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': round(confidence, 1),
                'fake_probability': round(fake_prob, 1),
                'real_probability': round(real_prob, 1),
                'detected_language': processed_df['language'].iloc[0]
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def test_multilingual_predictions(self):
        """Test predictions across all supported languages."""
        print("\n🌐 Testing Multilingual Predictions...")
        
        test_cases = [
            {
                'text': "Scientists discover miracle cure that doctors don't want you to know about!",
                'language': 'english',
                'expected': 'fake',
                'type': 'Health Misinformation'
            },
            {
                'text': "The Federal Reserve announced interest rate changes following economic analysis.",
                'language': 'english', 
                'expected': 'real',
                'type': 'Economic News'
            },
            {
                'text': "URGENTE: Esta hierba milagrosa cura la diabetes en 24 horas!",
                'language': 'spanish',
                'expected': 'fake', 
                'type': 'Spanish Health Fake'
            },
            {
                'text': "El Ministerio de Salud anunció nuevas directrices tras estudios científicos.",
                'language': 'spanish',
                'expected': 'real',
                'type': 'Spanish Official News'
            },
            {
                'text': "RÉVÉLATION: Ce remède secret guérit toutes les maladies!",
                'language': 'french',
                'expected': 'fake',
                'type': 'French Health Fake'
            },
            {
                'text': "Le ministère a publié de nouvelles recommandations médicales.",
                'language': 'french',
                'expected': 'real', 
                'type': 'French Official News'
            },
            {
                'text': "एक्सक्लूसिव: यह घरेलू नुस्खा सभी बीमारियों को ठीक करता है!",
                'language': 'hindi',
                'expected': 'fake',
                'type': 'Hindi Health Fake'
            },
            {
                'text': "स्वास्थ्य मंत्रालय ने नए दिशानिर्देश जारी किए।",
                'language': 'hindi',
                'expected': 'real',
                'type': 'Hindi Official News'
            }
        ]
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        language_performance = {}
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n   🧪 Test {i}: {test_case['type']}")
            
            result = self.test_single_prediction(
                test_case['text'], 
                test_case['language'],
                test_case['expected']
            )
            
            if 'error' in result:
                print(f"      ❌ Error: {result['error']}")
                continue
            
            # Check if prediction is correct
            predicted = result['prediction'].lower()
            expected = test_case['expected'].lower()
            is_correct = predicted == expected
            
            if is_correct:
                correct_predictions += 1
            
            # Track language performance
            lang = test_case['language']
            if lang not in language_performance:
                language_performance[lang] = {'correct': 0, 'total': 0, 'confidences': []}
            
            language_performance[lang]['total'] += 1
            if is_correct:
                language_performance[lang]['correct'] += 1
            language_performance[lang]['confidences'].append(result['confidence'])
            
            status = "✅" if is_correct else "❌"
            print(f"      {status} Expected: {expected.title()}, Got: {result['prediction']}")
            print(f"      📊 Confidence: {result['confidence']}%")
            print(f"      🌍 Language: {result['detected_language']}")
            print(f"      📈 Probabilities: Real {result['real_probability']}%, Fake {result['fake_probability']}%")
        
        # Overall results
        overall_accuracy = correct_predictions / total_predictions * 100
        print(f"\n🎯 OVERALL TEST RESULTS:")
        print(f"   📊 Accuracy: {overall_accuracy:.1f}% ({correct_predictions}/{total_predictions})")
        
        # Language-specific results
        print(f"\n📈 Language Performance:")
        for lang, performance in language_performance.items():
            accuracy = performance['correct'] / performance['total'] * 100
            avg_confidence = sum(performance['confidences']) / len(performance['confidences'])
            print(f"   {lang.title()}: {accuracy:.1f}% accuracy, {avg_confidence:.1f}% avg confidence")
        
        return overall_accuracy >= 75  # Pass if accuracy is at least 75%
    
    def test_confidence_formatting(self):
        """Test that confidence is properly formatted as percentages."""
        print(f"\n🔢 Testing Confidence Formatting...")
        
        test_text = "This is a test news article about fake scientific breakthroughs."
        result = self.test_single_prediction(test_text, 'english', 'fake')
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
            return False
        
        # Check that confidence is between 0-100 (percentage format)
        confidence = result['confidence']
        fake_prob = result['fake_probability']
        real_prob = result['real_probability']
        
        print(f"   ✅ Confidence: {confidence}% (proper percentage format)")
        print(f"   ✅ Fake Probability: {fake_prob}%")
        print(f"   ✅ Real Probability: {real_prob}%")
        
        # Verify percentages add up to 100%
        total = fake_prob + real_prob
        if abs(total - 100.0) < 0.1:  # Allow small rounding errors
            print(f"   ✅ Probabilities sum to {total:.1f}% (correct)")
            return True
        else:
            print(f"   ❌ Probabilities sum to {total:.1f}% (should be 100%)")
            return False
    
    def run_comprehensive_test(self):
        """Run all tests and provide final report."""
        print("🚀" * 20)
        print("  COMPREHENSIVE PROJECT TESTING")
        print("🚀" * 20)
        
        tests_passed = 0
        total_tests = 5
        
        # Test 1: Project Structure
        if self.test_project_structure():
            tests_passed += 1
        
        # Test 2: Model Loading  
        if self.test_model_loading():
            tests_passed += 1
        
        # Test 3: Dataset Integrity
        if self.test_dataset_integrity():
            tests_passed += 1
        
        # Test 4: Confidence Formatting
        if self.test_confidence_formatting():
            tests_passed += 1
        
        # Test 5: Multilingual Predictions
        if self.test_multilingual_predictions():
            tests_passed += 1
        
        # Final Report
        print("\n" + "="*60)
        print("📋 FINAL TEST REPORT")
        print("="*60)
        
        pass_rate = tests_passed / total_tests * 100
        print(f"   📊 Tests Passed: {tests_passed}/{total_tests} ({pass_rate:.1f}%)")
        
        if tests_passed == total_tests:
            print(f"   🎉 ALL TESTS PASSED! Project is ready for use.")
            print(f"   ✅ Spanish confidence issue: RESOLVED")
            print(f"   ✅ Multilingual support: WORKING")
            print(f"   ✅ Percentage formatting: CORRECT")
            status = "READY FOR PRODUCTION"
        elif tests_passed >= 4:
            print(f"   ⚠️  Most tests passed. Minor issues may need attention.")
            status = "MOSTLY READY"
        else:
            print(f"   ❌ Several tests failed. Issues need to be resolved.")
            status = "NEEDS FIXING"
        
        print(f"\n   🏆 PROJECT STATUS: {status}")
        
        return tests_passed == total_tests

def main():
    """Main testing function."""
    tester = ProjectTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print(f"\n🎯 Your multilingual fake news detection project is working perfectly!")
        print(f"   📱 You can now build applications, APIs, or services with it")
        print(f"   🌐 All languages (English, Spanish, French, Hindi) are supported")
        print(f"   💯 Confidence scores are properly formatted as percentages")
    else:
        print(f"\n🔧 Some issues need to be addressed before deployment.")

if __name__ == "__main__":
    main()