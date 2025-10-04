#!/usr/bin/env python3
"""
Enhanced Multilingual Preprocessing Module
Handles preprocessing for English, Spanish, French, and Hindi fake news detection
"""
import pandas as pd
import numpy as np
import re
import string
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import unicodedata
import warnings
warnings.filterwarnings('ignore')

class MultilingualPreprocessor:
    """
    Advanced multilingual text preprocessing for fake news detection.
    Supports English, Spanish, French, and Hindi.
    """
    
    def __init__(self, max_features=15000):
        self.max_features = max_features
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        
        # Download required NLTK data
        self._download_nltk_resources()
        
        # Initialize language-specific components
        self._initialize_language_components()
        
        # Language detection patterns
        self.language_patterns = {
            'spanish': [
                'el', 'la', 'los', 'las', 'de', 'del', 'que', 'y', 'en', 'un', 'una', 'con',
                'por', 'para', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'es', 'son', 'como',
                'pero', 'más', 'muy', 'también', 'puede', 'ser', 'hacer', 'tiene', 'todo'
            ],
            'french': [
                'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
                'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
                'par', 'grand', 'comme', 'mais', 'faire', 'bien', 'où', 'sans', 'peut'
            ],
            'hindi': [
                'है', 'में', 'की', 'को', 'का', 'से', 'और', 'यह', 'एक', 'के', 'लिए', 'पर',
                'था', 'हैं', 'थी', 'इस', 'तो', 'ही', 'भी', 'या', 'वह', 'जो', 'कि', 'गया',
                'दिया', 'किया', 'गई', 'हो', 'रहा', 'रही', 'होता', 'होती'
            ],
            'english': [
                'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was',
                'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this',
                'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what'
            ]
        }
    
    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        resources = [
            'punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4',
            'averaged_perceptron_tagger', 'universal_tagset'
        ]
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass
    
    def _initialize_language_components(self):
        """Initialize language-specific components."""
        # Stemmers for different languages
        self.stemmers = {
            'english': SnowballStemmer('english'),
            'spanish': SnowballStemmer('spanish'),
            'french': SnowballStemmer('french'),
            'hindi': None  # Hindi doesn't have a stemmer in NLTK
        }
        
        # Stopwords for different languages
        self.stop_words = {}
        for lang in ['english', 'spanish', 'french']:
            try:
                self.stop_words[lang] = set(stopwords.words(lang))
            except:
                self.stop_words[lang] = set()
        
        # Hindi stopwords (manual list since NLTK doesn't have them)
        self.stop_words['hindi'] = {
            'है', 'में', 'की', 'को', 'का', 'से', 'और', 'यह', 'एक', 'के', 'लिए', 'पर',
            'था', 'हैं', 'थी', 'इस', 'तो', 'ही', 'भी', 'या', 'वह', 'जो', 'कि', 'गया',
            'दिया', 'किया', 'गई', 'हो', 'रहा', 'रही', 'होता', 'होती', 'करने', 'होने',
            'वाले', 'वाली', 'वाला', 'कर', 'कहा', 'कहते', 'बात', 'लोग', 'आप', 'हम',
            'तुम', 'अब', 'यहाँ', 'वहाँ', 'कैसे', 'क्यों', 'कब', 'कहाँ', 'कौन', 'क्या'
        }
        
        # Lemmatizer (works mainly for English)
        self.lemmatizer = WordNetLemmatizer()
    
    def detect_language(self, text):
        """Detect the language of the text."""
        if pd.isna(text) or not isinstance(text, str):
            return 'english'  # default
        
        text_lower = text.lower()
        scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            scores[lang] = score / len(patterns)  # Normalize by pattern count
        
        # Return language with highest score
        detected_lang = max(scores, key=scores.get)
        
        # If score is too low, default to English
        if scores[detected_lang] < 0.01:
            return 'english'
        
        return detected_lang
    
    def clean_text(self, text):
        """Basic text cleaning."""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep unicode for non-English languages
        text = re.sub(r'[^\w\s\u0900-\u097F\u00C0-\u00FF\u0100-\u017F\u1E00-\u1EFF]', ' ', text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        return text.strip()
    
    def preprocess_by_language(self, text, language):
        """Language-specific preprocessing."""
        if not text:
            return ""
        
        # Clean text first
        text = self.clean_text(text)
        
        if language == 'hindi':
            return self._preprocess_hindi(text)
        elif language == 'spanish':
            return self._preprocess_spanish(text)
        elif language == 'french':
            return self._preprocess_french(text)
        else:  # English or unknown
            return self._preprocess_english(text)
    
    def _preprocess_english(self, text):
        """English-specific preprocessing."""
        # Tokenize
        try:
            tokens = word_tokenize(text.lower())
        except:
            tokens = text.lower().split()
        
        # Remove punctuation and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words['english']]
        
        # Lemmatize
        try:
            # Get POS tags for better lemmatization
            pos_tags = pos_tag(tokens)
            lemmatized_tokens = []
            
            for token, pos in pos_tags:
                # Convert POS tag to wordnet format
                if pos.startswith('J'):
                    pos_tag_wn = 'a'  # adjective
                elif pos.startswith('V'):
                    pos_tag_wn = 'v'  # verb
                elif pos.startswith('N'):
                    pos_tag_wn = 'n'  # noun
                elif pos.startswith('R'):
                    pos_tag_wn = 'r'  # adverb
                else:
                    pos_tag_wn = 'n'  # default to noun
                
                lemmatized_tokens.append(self.lemmatizer.lemmatize(token, pos=pos_tag_wn))
            
            tokens = lemmatized_tokens
        except:
            # Fallback to simple lemmatization
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
    
    def _preprocess_spanish(self, text):
        """Spanish-specific preprocessing."""
        # Tokenize
        try:
            tokens = word_tokenize(text.lower())
        except:
            tokens = text.lower().split()
        
        # Remove punctuation and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words['spanish']]
        
        # Stem (Spanish stemming)
        if self.stemmers['spanish']:
            tokens = [self.stemmers['spanish'].stem(token) for token in tokens]
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
    
    def _preprocess_french(self, text):
        """French-specific preprocessing."""
        # Tokenize
        try:
            tokens = word_tokenize(text.lower())
        except:
            tokens = text.lower().split()
        
        # Remove punctuation and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words['french']]
        
        # Stem (French stemming)
        if self.stemmers['french']:
            tokens = [self.stemmers['french'].stem(token) for token in tokens]
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
    
    def _preprocess_hindi(self, text):
        """Hindi-specific preprocessing."""
        # Basic tokenization (split by whitespace for Hindi)
        tokens = text.split()
        
        # Convert to lowercase (for consistency)
        tokens = [token.lower() for token in tokens]
        
        # Remove punctuation and non-Devanagari/Latin characters
        tokens = [re.sub(r'[^\u0900-\u097F\w]', '', token) for token in tokens]
        
        # Remove empty tokens
        tokens = [token for token in tokens if token]
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words['hindi']]
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return ' '.join(tokens)
    
    def preprocess_dataset(self, df):
        """Preprocess the entire dataset with multilingual support."""
        print("🔧 Starting multilingual preprocessing...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Detect language if not provided
        if 'language' not in processed_df.columns:
            print("   🔍 Detecting languages...")
            processed_df['language'] = processed_df['text'].apply(self.detect_language)
        
        # Show language distribution
        lang_counts = processed_df['language'].value_counts()
        print(f"   📊 Language distribution:")
        for lang, count in lang_counts.items():
            percentage = count / len(processed_df) * 100
            print(f"      {lang.title()}: {count:,} articles ({percentage:.1f}%)")
        
        # Preprocess text by language
        print("   🔄 Processing texts by language...")
        processed_texts = []
        
        for idx, row in processed_df.iterrows():
            if idx % 5000 == 0 and idx > 0:
                print(f"      Processed {idx:,}/{len(processed_df):,} articles...")
            
            text = row['text']
            language = row.get('language', 'english')
            
            processed_text = self.preprocess_by_language(text, language)
            processed_texts.append(processed_text)
        
        processed_df['processed_text'] = processed_texts
        
        # Remove empty processed texts
        initial_count = len(processed_df)
        processed_df = processed_df[processed_df['processed_text'].str.len() > 0]
        removed_count = initial_count - len(processed_df)
        
        if removed_count > 0:
            print(f"   🗑️  Removed {removed_count} articles with empty processed text")
        
        print(f"   ✅ Preprocessing complete!")
        print(f"      📊 Final dataset size: {len(processed_df):,} articles")
        
        return processed_df
    
    def vectorize_texts(self, processed_texts, fit=True):
        """Vectorize processed texts using TF-IDF."""
        print("🔢 Vectorizing texts...")
        
        if fit:
            # Initialize vectorizer with multilingual support
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),  # Include bigrams
                min_df=2,  # Minimum document frequency
                max_df=0.8,  # Maximum document frequency
                lowercase=True,
                stop_words=None,  # Already removed language-specific stopwords
                sublinear_tf=True,  # Apply sublinear tf scaling
                smooth_idf=True
            )
            
            # Fit and transform
            X = self.vectorizer.fit_transform(processed_texts)
        else:
            if self.vectorizer is None:
                raise ValueError("Vectorizer not fitted yet!")
            
            # Transform only
            X = self.vectorizer.transform(processed_texts)
        
        print(f"   ✅ Vectorization complete!")
        print(f"      📊 Feature matrix shape: {X.shape}")
        print(f"      🔢 Features: {self.vectorizer.get_feature_names_out()[:10]}...")
        
        return X
    
    def process_and_vectorize(self, df, fit=True):
        """Complete preprocessing and vectorization pipeline."""
        print("🚀" * 15)
        print("  MULTILINGUAL PREPROCESSING PIPELINE")
        print("🚀" * 15)
        
        # Preprocess dataset
        processed_df = self.preprocess_dataset(df)
        
        # Vectorize texts
        X = self.vectorize_texts(processed_df['processed_text'], fit=fit)
        
        # Encode labels
        if fit:
            y = self.label_encoder.fit_transform(processed_df['label'])
        else:
            y = self.label_encoder.transform(processed_df['label'])
        
        print(f"\n✅ Complete preprocessing pipeline finished!")
        print(f"   📊 Dataset: {len(processed_df):,} articles")
        print(f"   🌍 Languages: {', '.join(processed_df['language'].unique())}")
        print(f"   🔢 Features: {X.shape[1]:,}")
        print(f"   📈 Ready for multilingual training!")
        
        return X, y, processed_df

def test_multilingual_preprocessing():
    """Test the multilingual preprocessing with sample data."""
    print("🧪 Testing multilingual preprocessing...")
    
    # Sample multilingual data
    test_data = {
        'text': [
            "This is fake news about health scams",  # English
            "Esta es una noticia falsa sobre estafas de salud",  # Spanish  
            "Ceci est une fausse nouvelle sur les arnaques de santé",  # French
            "यह स्वास्थ्य घोटालों के बारे में झूठी खबर है"  # Hindi
        ],
        'label': [1, 1, 1, 1],
        'language': ['english', 'spanish', 'french', 'hindi']
    }
    
    df = pd.DataFrame(test_data)
    
    # Initialize preprocessor
    preprocessor = MultilingualPreprocessor(max_features=1000)
    
    # Process
    X, y, processed_df = preprocessor.process_and_vectorize(df)
    
    print(f"✅ Test successful!")
    print(f"   Processed texts: {list(processed_df['processed_text'])}")
    print(f"   Feature matrix shape: {X.shape}")

if __name__ == "__main__":
    test_multilingual_preprocessing()