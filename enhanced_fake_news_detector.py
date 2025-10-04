#!/usr/bin/env python3
"""
Enhanced Fake News Detector with Web Search Fact Verification
Combines ML-based pattern detection with real-time fact checking
"""
import os
import sys
import re
import requests
from bs4 import BeautifulSoup
import time
import warnings
from urllib.parse import urlparse
import json

warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.large_scale_preprocess import LargeScaleNewsPreprocessor
from src.model import FakeNewsClassifier
import joblib

class FactChecker:
    """
    Web-based fact checker that verifies claims using search engines and reliable sources.
    """
    
    def __init__(self):
        self.reliable_domains = {
            'wikipedia.org', 'bbc.com', 'reuters.com', 'ap.org', 'cnn.com',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com', 'npr.org',
            'espn.com', 'cricinfo.com', 'who.int', 'cdc.gov', 'nih.gov',
            'nature.com', 'science.org', 'pubmed.ncbi.nlm.nih.gov'
        }
        
    def extract_key_claims(self, text):
        """Extract key factual claims from text."""
        # Simple pattern matching for common claim types
        patterns = [
            r'(.+?)\s+has more\s+(.+?)\s+than\s+(.+)',  # "X has more Y than Z"
            r'(.+?)\s+is\s+a\s+(.+)',  # "X is a Y"
            r'(.+?)\s+scored\s+(\d+)\s+(.+)',  # "X scored N Y"
            r'(.+?)\s+won\s+(\d+)\s+(.+)',  # "X won N Y"
            r'(.+?)\s+was born in\s+(.+)',  # "X was born in Y"
            r'(.+?)\s+plays for\s+(.+)',  # "X plays for Y"
        ]
        
        claims = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append(' '.join(match))
        
        if not claims:
            # If no specific pattern, use the whole text as a claim
            claims.append(text.strip())
            
        return claims
    
    def search_web_simple(self, query, num_results=3):
        """Simple web search using a basic approach."""
        try:
            # Use DuckDuckGo instant answer API (no rate limits)
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1&skip_disambig=1"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for instant answer
                if data.get('Answer'):
                    return [{'title': 'DuckDuckGo Answer', 'snippet': data['Answer'], 'url': 'duckduckgo.com'}]
                
                # Check for abstract
                if data.get('Abstract'):
                    return [{'title': data.get('AbstractSource', 'Source'), 'snippet': data['Abstract'], 'url': data.get('AbstractURL', '')}]
                
                # Check for infobox
                if data.get('Infobox'):
                    content = data['Infobox'].get('content', [])
                    if content:
                        info = ' '.join([item.get('value', '') for item in content[:3]])
                        return [{'title': 'Infobox Info', 'snippet': info, 'url': 'various'}]
                        
            return []
            
        except Exception as e:
            print(f"⚠️ Web search failed: {e}")
            return []
    
    def analyze_search_results(self, results, original_claim):
        """Analyze search results to determine claim validity."""
        if not results:
            return {"confidence": 0.5, "verdict": "unknown", "evidence": "No search results found"}
        
        evidence_text = ""
        reliable_sources = 0
        total_sources = len(results)
        
        for result in results:
            evidence_text += result.get('snippet', '') + " "
            
            # Check if source is reliable
            url = result.get('url', '')
            domain = urlparse(url).netloc.lower() if url else ''
            
            if any(reliable in domain for reliable in self.reliable_domains):
                reliable_sources += 1
        
        # Simple contradiction detection
        original_lower = original_claim.lower()
        evidence_lower = evidence_text.lower()
        
        # Extract key terms from original claim
        claim_words = set(re.findall(r'\b\w+\b', original_lower))
        evidence_words = set(re.findall(r'\b\w+\b', evidence_lower))
        
        # Check for contradictory terms
        contradiction_indicators = [
            'not', 'never', 'false', 'incorrect', 'wrong', 'myth', 'debunked',
            'actually', 'however', 'but', 'although', 'contrary'
        ]
        
        contradiction_count = sum(1 for word in contradiction_indicators if word in evidence_lower)
        word_overlap = len(claim_words.intersection(evidence_words))
        
        # Calculate confidence based on multiple factors
        base_confidence = min(0.8, word_overlap / max(len(claim_words), 1) * 0.7)
        reliability_bonus = (reliable_sources / total_sources) * 0.2
        
        if contradiction_count > 0:
            # Likely contradicted by evidence
            confidence = base_confidence + reliability_bonus
            verdict = "likely_false"
        elif word_overlap > len(claim_words) * 0.5:
            # Good overlap with evidence
            confidence = base_confidence + reliability_bonus
            verdict = "likely_true"
        else:
            # Insufficient evidence
            confidence = 0.5
            verdict = "unknown"
        
        return {
            "confidence": min(confidence, 0.95),  # Cap at 95%
            "verdict": verdict,
            "evidence": evidence_text[:200] + "..." if len(evidence_text) > 200 else evidence_text,
            "reliable_sources": reliable_sources,
            "total_sources": total_sources
        }
    
    def verify_claim(self, claim):
        """Verify a single claim using web search."""
        print(f"🔍 Fact-checking: {claim}")
        
        # Search for the claim
        results = self.search_web_simple(claim)
        
        # Analyze results
        analysis = self.analyze_search_results(results, claim)
        
        return analysis

class EnhancedFakeNewsDetector:
    """
    Enhanced fake news detector combining ML patterns with fact verification.
    """
    
    def __init__(self, model_dir='models_large_scale', enable_fact_check=True):
        self.model_dir = model_dir
        self.enable_fact_check = enable_fact_check
        self.preprocessor = LargeScaleNewsPreprocessor()
        self.classifier = FakeNewsClassifier()
        self.fact_checker = FactChecker() if enable_fact_check else None
        self.model_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load ML models."""
        try:
            # Load vectorizer
            vectorizer_path = os.path.join(self.model_dir, 'large_scale_vectorizer.pkl')
            if os.path.exists(vectorizer_path):
                self.preprocessor.load_vectorizer(vectorizer_path)
                print(f"✅ Loaded ML vectorizer")
            else:
                print(f"❌ Vectorizer not found")
                return False
            
            # Load best model
            best_model_files = [f for f in os.listdir(self.model_dir) 
                              if f.startswith('best_model_large_scale_') and f.endswith('.pkl')]
            
            if best_model_files:
                best_model_path = os.path.join(self.model_dir, best_model_files[0])
                model_name = best_model_files[0].replace('best_model_large_scale_', '').replace('.pkl', '')
                
                model = joblib.load(best_model_path)
                self.classifier.trained_models[model_name] = model
                self.classifier.best_model = model
                self.classifier.best_model_name = model_name
                
                print(f"✅ Loaded ML model: {model_name}")
                self.model_loaded = True
                return True
            else:
                print(f"❌ No ML model found")
                return False
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def analyze_with_ml(self, text):
        """Analyze text using ML model for pattern detection."""
        if not self.model_loaded:
            return {"error": "ML models not loaded"}
        
        try:
            processed_text = self.preprocessor.preprocess_text_parallel(text)
            text_tfidf = self.preprocessor.vectorizer.transform([processed_text])
            
            prediction = self.classifier.best_model.predict(text_tfidf)[0]
            
            try:
                probabilities = self.classifier.best_model.predict_proba(text_tfidf)[0]
                real_prob = probabilities[0]
                fake_prob = probabilities[1]
                ml_confidence = max(real_prob, fake_prob)
            except:
                real_prob = fake_prob = 0.5
                ml_confidence = 0.5
            
            return {
                "prediction": prediction,
                "confidence": ml_confidence,
                "probabilities": {"real": real_prob, "fake": fake_prob}
            }
            
        except Exception as e:
            return {"error": f"ML analysis failed: {e}"}
    
    def comprehensive_analysis(self, text):
        """Perform comprehensive analysis combining ML and fact-checking."""
        print(f"\n🔬 Analyzing: {text}\n")
        
        # Step 1: ML-based pattern analysis
        print("📊 Step 1: ML Pattern Analysis")
        ml_result = self.analyze_with_ml(text)
        
        if "error" in ml_result:
            return ml_result
        
        ml_prediction = "FAKE" if ml_result["prediction"] == 1 else "REAL"
        ml_confidence = ml_result["confidence"]
        
        print(f"   ML Prediction: {ml_prediction} ({ml_confidence:.1%} confidence)")
        
        # Step 2: Fact verification (if enabled)
        fact_results = []
        if self.enable_fact_check and self.fact_checker:
            print("\n🔍 Step 2: Fact Verification")
            claims = self.fact_checker.extract_key_claims(text)
            
            for claim in claims[:2]:  # Limit to 2 claims to avoid rate limiting
                fact_result = self.fact_checker.verify_claim(claim)
                fact_results.append(fact_result)
                
                verdict_emoji = {"likely_true": "✅", "likely_false": "❌", "unknown": "❓"}
                verdict = fact_result["verdict"]
                print(f"   {verdict_emoji.get(verdict, '❓')} {verdict.replace('_', ' ').title()}: {fact_result['confidence']:.1%} confidence")
                if fact_result['evidence']:
                    print(f"   📝 Evidence: {fact_result['evidence'][:100]}...")
        
        # Step 3: Combined analysis
        print(f"\n🎯 Step 3: Combined Analysis")
        
        final_prediction, final_confidence, reasoning = self.combine_results(
            ml_result, fact_results, text
        )
        
        return {
            "text": text,
            "final_prediction": final_prediction,
            "final_confidence": final_confidence,
            "reasoning": reasoning,
            "ml_analysis": ml_result,
            "fact_checking": fact_results
        }
    
    def combine_results(self, ml_result, fact_results, original_text):
        """Combine ML and fact-checking results for final prediction."""
        ml_prediction = ml_result["prediction"]
        ml_confidence = ml_result["confidence"]
        
        # If no fact-checking results, use ML only
        if not fact_results:
            prediction = "FAKE NEWS" if ml_prediction == 1 else "REAL NEWS"
            return prediction, ml_confidence, "Based on ML pattern analysis only"
        
        # Analyze fact-checking results
        false_facts = sum(1 for fr in fact_results if fr["verdict"] == "likely_false")
        true_facts = sum(1 for fr in fact_results if fr["verdict"] == "likely_true")
        unknown_facts = sum(1 for fr in fact_results if fr["verdict"] == "unknown")
        
        avg_fact_confidence = sum(fr["confidence"] for fr in fact_results) / len(fact_results)
        
        # Decision logic
        reasoning_parts = []
        
        if false_facts > 0:
            # If any facts are likely false, lean towards FAKE
            fact_weight = 0.7
            ml_weight = 0.3
            
            if ml_prediction == 1:  # ML also says fake
                final_confidence = fact_weight * avg_fact_confidence + ml_weight * ml_confidence
                prediction = "FAKE NEWS"
                reasoning_parts.append(f"Contains {false_facts} likely false claim(s)")
                reasoning_parts.append("ML pattern analysis agrees")
            else:  # ML says real but facts are false
                final_confidence = fact_weight * avg_fact_confidence + ml_weight * (1 - ml_confidence)
                prediction = "FAKE NEWS"
                reasoning_parts.append(f"Contains {false_facts} factually incorrect claim(s)")
                reasoning_parts.append("Overrides ML pattern analysis")
                
        elif true_facts > unknown_facts:
            # If mostly true facts, lean towards REAL
            fact_weight = 0.6
            ml_weight = 0.4
            
            final_confidence = fact_weight * avg_fact_confidence + ml_weight * ml_confidence
            if ml_prediction == 0:  # ML also says real
                prediction = "REAL NEWS"
                reasoning_parts.append("Facts check out as likely true")
                reasoning_parts.append("ML pattern analysis agrees")
            else:  # Facts true but ML suspicious
                prediction = "REAL NEWS" if avg_fact_confidence > 0.7 else "UNCERTAIN"
                reasoning_parts.append("Facts appear correct")
                reasoning_parts.append("ML detected suspicious patterns")
                
        else:
            # Mostly unknown facts, rely more on ML
            fact_weight = 0.3
            ml_weight = 0.7
            
            final_confidence = ml_weight * ml_confidence + fact_weight * avg_fact_confidence
            prediction = "FAKE NEWS" if ml_prediction == 1 else "REAL NEWS"
            reasoning_parts.append("Insufficient fact verification")
            reasoning_parts.append("Based primarily on ML analysis")
        
        reasoning = "; ".join(reasoning_parts)
        return prediction, min(final_confidence, 0.95), reasoning

def interactive_enhanced_test():
    """Interactive testing with enhanced capabilities."""
    
    print("🚀" * 20)
    print("  ENHANCED FAKE NEWS DETECTOR WITH FACT CHECKING")
    print("🚀" * 20)
    print("This system combines ML pattern analysis with real-time fact verification!")
    print("Type 'quit' to exit\n")
    
    detector = EnhancedFakeNewsDetector(enable_fact_check=True)
    
    if not detector.model_loaded:
        print("❌ Could not load ML models. Please train first.")
        return
    
    while True:
        try:
            text = input("📰 Enter news text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            # Comprehensive analysis
            result = detector.comprehensive_analysis(text)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Display results
            print("\n" + "=" * 60)
            print(f"🤖 FINAL PREDICTION: {result['final_prediction']}")
            print(f"📊 CONFIDENCE: {result['final_confidence']:.1%}")
            print(f"🧠 REASONING: {result['reasoning']}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_enhanced_test()