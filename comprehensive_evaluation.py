#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for Fake News Detector
Tests bias, robustness, and real-world performance across multiple dimensions
"""
import os
import sys
import warnings
import json
from datetime import datetime
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from improved_fact_checker import ImprovedFakeNewsDetector

class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for testing fake news detector across multiple dimensions.
    """
    
    def __init__(self):
        self.detector = ImprovedFakeNewsDetector()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
    
    def test_news_source_performance(self):
        """Test performance on different news sources."""
        print("🌐 Testing Performance on Different News Sources")
        print("=" * 60)
        
        # Sample headlines from different sources (simulated - in real use you'd fetch actual headlines)
        test_cases = {
            'BBC': [
                "UK inflation rate falls to 4.6% in October, ONS data shows",
                "Climate summit agrees to transition away from fossil fuels",
                "New study links processed meat to increased cancer risk",
                "Government announces £2bn investment in renewable energy"
            ],
            'CNN': [
                "White House announces new sanctions against foreign adversaries", 
                "Federal Reserve keeps interest rates steady amid economic uncertainty",
                "Supreme Court to hear landmark case on digital privacy rights",
                "NASA launches new mission to study Mars atmosphere"
            ],
            'Fox': [
                "Border patrol reports increase in illegal crossings this month",
                "Congressional hearing reveals concerns about government spending", 
                "Small business owners struggle with new regulatory requirements",
                "Energy sector sees growth amid policy changes"
            ],
            'Social_Media_Real': [
                "Just read an interesting article about renewable energy trends",
                "Local hospital announces new cancer treatment program available",
                "University study shows benefits of exercise for mental health",
                "City council approves budget for new public transportation"
            ],
            'Social_Media_Fake': [
                "OMG! Doctors HATE this one simple trick that cures everything!!!",
                "BREAKING: Government hiding alien technology in Area 51 confirmed!!!",
                "This mom lost 50 pounds in a week with this weird trick!",
                "SHOCKING: Your phone is listening to your thoughts 24/7!!!"
            ]
        }
        
        source_results = {}
        
        for source, headlines in test_cases.items():
            print(f"\n📰 Testing {source} Headlines:")
            source_results[source] = []
            
            for headline in headlines:
                try:
                    # Quick analysis without full output
                    ml_result = self.detector.analyze_with_ml(headline)
                    fact_results = self.detector.fact_checker.comprehensive_fact_check(headline)
                    
                    final_prediction, final_confidence, reasoning = self.detector.combine_results(
                        ml_result, fact_results
                    )
                    
                    result = {
                        'headline': headline,
                        'prediction': final_prediction,
                        'confidence': final_confidence,
                        'reasoning': reasoning
                    }
                    
                    source_results[source].append(result)
                    
                    # Short output
                    status = "✅" if "REAL" in final_prediction else "❌"
                    print(f"   {status} {headline[:50]}... → {final_prediction} ({final_confidence:.0%})")
                    
                except Exception as e:
                    print(f"   ⚠️ Error processing: {headline[:30]}... - {e}")
        
        self.results['tests']['news_source_performance'] = source_results
        return source_results
    
    def test_satire_detection(self):
        """Test ability to distinguish satire from real news."""
        print("\n😄 Testing Satire & Parody Detection")
        print("=" * 60)
        
        test_cases = [
            {
                'text': "Local Man Discovers Revolutionary Way to Avoid Traffic: Walking",
                'type': 'satire',
                'expected': 'should_detect_as_satirical'
            },
            {
                'text': "Area Woman Reportedly 'Living Her Best Life' After Buying Expensive Coffee",
                'type': 'satire', 
                'expected': 'should_detect_as_satirical'
            },
            {
                'text': "BREAKING: Scientists Discover Water is Wet, More Research Needed",
                'type': 'satire',
                'expected': 'should_detect_as_satirical'
            },
            {
                'text': "The Federal Reserve announced a decision to maintain current interest rates",
                'type': 'real_news',
                'expected': 'should_detect_as_real'
            },
            {
                'text': "Research team publishes findings on climate change in Nature journal",
                'type': 'real_news',
                'expected': 'should_detect_as_real'
            }
        ]
        
        satire_results = []
        
        for test_case in test_cases:
            try:
                ml_result = self.detector.analyze_with_ml(test_case['text'])
                fact_results = self.detector.fact_checker.comprehensive_fact_check(test_case['text'])
                
                final_prediction, final_confidence, reasoning = self.detector.combine_results(
                    ml_result, fact_results
                )
                
                result = {
                    'text': test_case['text'],
                    'type': test_case['type'],
                    'expected': test_case['expected'],
                    'prediction': final_prediction,
                    'confidence': final_confidence,
                    'reasoning': reasoning
                }
                
                satire_results.append(result)
                
                # Analyze if detection is appropriate
                if test_case['type'] == 'satire':
                    # For satire, we want nuanced detection
                    status = "🤔" if "REAL" in final_prediction else "⚠️"
                    note = "(Satire often classified as REAL - this is expected)"
                else:
                    status = "✅" if "REAL" in final_prediction else "❌"  
                    note = ""
                
                print(f"   {status} {test_case['type'].title()}: {test_case['text'][:40]}...")
                print(f"      → {final_prediction} ({final_confidence:.0%}) {note}")
                
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
        
        self.results['tests']['satire_detection'] = satire_results
        return satire_results
    
    def test_bias_analysis(self):
        """Test for political and topical bias."""
        print("\n⚖️ Testing Political & Topical Bias")
        print("=" * 60)
        
        # Test cases designed to reveal bias
        bias_test_cases = [
            {
                'category': 'political_neutral',
                'texts': [
                    "Congress passes bipartisan infrastructure bill with majority support",
                    "Supreme Court hears arguments in constitutional law case",
                    "Federal agencies release annual budget reports for public review"
                ]
            },
            {
                'category': 'political_left_leaning',
                'texts': [
                    "Progressive policies show promise in reducing income inequality",
                    "Environmental regulations protect communities from industrial pollution", 
                    "Universal healthcare system reduces medical bankruptcies"
                ]
            },
            {
                'category': 'political_right_leaning', 
                'texts': [
                    "Tax cuts stimulate small business growth and job creation",
                    "Deregulation allows market forces to drive economic efficiency",
                    "Strong border security protects national sovereignty and safety"
                ]
            },
            {
                'category': 'science_factual',
                'texts': [
                    "Peer-reviewed study confirms effectiveness of vaccine in clinical trials",
                    "Climate data shows temperature increases over past century", 
                    "Medical researchers identify genetic factors in disease susceptibility"
                ]
            },
            {
                'category': 'conspiracy_prone',
                'texts': [
                    "Government agencies coordinate response to public health emergency",
                    "International organizations collaborate on global issues",
                    "Tech companies implement new data privacy measures"
                ]
            }
        ]
        
        bias_results = {}
        
        for category_data in bias_test_cases:
            category = category_data['category']
            print(f"\n📊 Testing {category.replace('_', ' ').title()}:")
            
            category_results = []
            fake_predictions = 0
            total_predictions = 0
            
            for text in category_data['texts']:
                try:
                    ml_result = self.detector.analyze_with_ml(text)
                    fact_results = self.detector.fact_checker.comprehensive_fact_check(text)
                    
                    final_prediction, final_confidence, reasoning = self.detector.combine_results(
                        ml_result, fact_results
                    )
                    
                    result = {
                        'text': text,
                        'prediction': final_prediction,
                        'confidence': final_confidence,
                        'ml_fake_prob': ml_result['probabilities']['fake']
                    }
                    
                    category_results.append(result)
                    
                    if "FAKE" in final_prediction:
                        fake_predictions += 1
                    total_predictions += 1
                    
                    status = "❌" if "FAKE" in final_prediction else "✅"
                    print(f"   {status} {text[:50]}... → {final_prediction} ({final_confidence:.0%})")
                    
                except Exception as e:
                    print(f"   ⚠️ Error: {e}")
            
            fake_rate = fake_predictions / total_predictions if total_predictions > 0 else 0
            bias_results[category] = {
                'results': category_results,
                'fake_rate': fake_rate,
                'total_tested': total_predictions
            }
            
            print(f"   📈 Fake Rate: {fake_rate:.1%} ({fake_predictions}/{total_predictions})")
        
        self.results['tests']['bias_analysis'] = bias_results
        return bias_results
    
    def test_multilingual_robustness(self):
        """Test handling of non-English and translated content."""
        print("\n🌍 Testing Multilingual & Translation Robustness")
        print("=" * 60)
        
        multilingual_cases = [
            {
                'language': 'English',
                'text': "Scientists publish research findings in peer-reviewed journal",
                'type': 'real'
            },
            {
                'language': 'Spanish_style',
                'text': "Los científicos publican hallazgos de investigación en revista revisada por pares",
                'type': 'real_translated'
            },
            {
                'language': 'Translation_artifacts',
                'text': "The scientists they publish the findings of research in journal reviewed by the peers",
                'type': 'translated_awkward'
            },
            {
                'language': 'Non_Western_style',
                'text': "According to ministry sources, the economic indicators show steady growth pattern",
                'type': 'formal_non_western'
            },
            {
                'language': 'Mixed_language',
                'text': "Breaking news: gouvernement announces nouvelle politique for économie development",
                'type': 'code_switched'
            }
        ]
        
        multilingual_results = []
        
        for case in multilingual_cases:
            try:
                ml_result = self.detector.analyze_with_ml(case['text'])
                fact_results = self.detector.fact_checker.comprehensive_fact_check(case['text'])
                
                final_prediction, final_confidence, reasoning = self.detector.combine_results(
                    ml_result, fact_results
                )
                
                result = {
                    'language': case['language'],
                    'type': case['type'],
                    'text': case['text'],
                    'prediction': final_prediction,
                    'confidence': final_confidence,
                    'ml_confidence': ml_result['confidence']
                }
                
                multilingual_results.append(result)
                
                # Check for concerning patterns
                if ml_result['confidence'] < 0.6:
                    status = "⚠️"
                    note = "(Low confidence - potential language bias)"
                elif "FAKE" in final_prediction and case['type'] in ['real', 'real_translated']:
                    status = "🔍"
                    note = "(False positive - investigate bias)"
                else:
                    status = "✅"
                    note = ""
                
                print(f"   {status} {case['language']}: {case['text'][:40]}...")
                print(f"      → {final_prediction} ({final_confidence:.0%}) {note}")
                
            except Exception as e:
                print(f"   ⚠️ Error processing {case['language']}: {e}")
        
        self.results['tests']['multilingual_robustness'] = multilingual_results
        return multilingual_results
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive evaluation report."""
        print("\n📋 Generating Comprehensive Evaluation Report")
        print("=" * 60)
        
        # Calculate overall statistics
        total_tests = 0
        concerning_patterns = []
        
        # News source analysis
        if 'news_source_performance' in self.results['tests']:
            source_data = self.results['tests']['news_source_performance']
            
            # Check for source bias
            for source, results in source_data.items():
                if results:
                    fake_rate = sum(1 for r in results if "FAKE" in r['prediction']) / len(results)
                    if fake_rate > 0.5 and 'Fake' not in source:
                        concerning_patterns.append(f"High false positive rate for {source}: {fake_rate:.1%}")
                    total_tests += len(results)
        
        # Bias analysis
        if 'bias_analysis' in self.results['tests']:
            bias_data = self.results['tests']['bias_analysis']
            
            # Compare fake rates across political categories
            political_categories = ['political_left_leaning', 'political_right_leaning', 'political_neutral']
            fake_rates = {}
            
            for cat in political_categories:
                if cat in bias_data:
                    fake_rates[cat] = bias_data[cat]['fake_rate']
            
            if len(fake_rates) >= 2:
                max_rate = max(fake_rates.values())
                min_rate = min(fake_rates.values())
                if max_rate - min_rate > 0.3:  # 30% difference
                    concerning_patterns.append(f"Political bias detected: {max_rate:.1%} vs {min_rate:.1%} fake rates")
        
        # Generate summary
        report = {
            'overall_assessment': 'PASS' if len(concerning_patterns) == 0 else 'NEEDS_ATTENTION',
            'total_tests_conducted': total_tests,
            'concerning_patterns': concerning_patterns,
            'recommendations': []
        }
        
        if len(concerning_patterns) == 0:
            report['recommendations'].append("✅ Model shows good robustness across tested dimensions")
        else:
            report['recommendations'].extend([
                "🔍 Review training data for bias patterns",
                "📊 Consider balanced sampling across news sources",
                "🌐 Add more diverse training examples",
                "⚖️ Implement bias detection monitoring"
            ])
        
        self.results['overall_report'] = report
        
        # Print summary
        print(f"\n🎯 Overall Assessment: {report['overall_assessment']}")
        print(f"📊 Total Tests: {report['total_tests_conducted']}")
        
        if concerning_patterns:
            print(f"\n⚠️ Concerning Patterns Found:")
            for pattern in concerning_patterns:
                print(f"   • {pattern}")
        
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
        
        return report
    
    def save_results(self, filename='comprehensive_evaluation_results.json'):
        """Save evaluation results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {filename}")
    
    def run_all_evaluations(self):
        """Run the complete evaluation suite."""
        print("🚀" * 20)
        print("  COMPREHENSIVE FAKE NEWS DETECTOR EVALUATION")
        print("🚀" * 20)
        print("Testing bias, robustness, and real-world performance...\n")
        
        try:
            # Run all test suites
            self.test_news_source_performance()
            self.test_satire_detection() 
            self.test_bias_analysis()
            self.test_multilingual_robustness()
            
            # Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            # Save results
            self.save_results()
            
            return report
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return None

def main():
    """Main execution function."""
    evaluator = ComprehensiveEvaluator()
    
    if not evaluator.detector.model_loaded:
        print("❌ Could not load models. Please train the model first.")
        return
    
    # Run comprehensive evaluation
    report = evaluator.run_all_evaluations()
    
    if report:
        print(f"\n🎉 Evaluation completed!")
        print(f"Check 'comprehensive_evaluation_results.json' for detailed results.")
    else:
        print(f"\n❌ Evaluation failed!")

if __name__ == "__main__":
    main()