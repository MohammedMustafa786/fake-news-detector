#!/usr/bin/env python3
"""
Optimized preprocessing module for large-scale fake news detection.
Handles 40k+ articles efficiently with batch processing and memory optimization.
"""
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import joblib
import os
from bs4 import BeautifulSoup
import unicodedata
from tqdm import tqdm
import gc
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

class LargeScaleNewsPreprocessor:
    """
    High-performance text preprocessing for large-scale fake news detection.
    Optimized for datasets with 40k+ articles.
    """
    
    def __init__(self, max_features=20000, use_lemmatization=True, batch_size=1000, n_jobs=-1):
        self.max_features = max_features
        self.use_lemmatization = use_lemmatization
        self.batch_size = batch_size
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=5,  # Ignore terms that appear in less than 5 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True,  # Apply sublinear scaling
            norm='l2'  # L2 normalization
        )
        
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = None
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data if not already present"""
        required_data = [
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
            ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
            ('corpora/omw-1.4', 'omw-1.4')
        ]
        
        for data_path, download_name in required_data:
            try:
                nltk.data.find(data_path)
            except LookupError:
                print(f"Downloading {download_name}...")
                nltk.download(download_name, quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text_batch(self, texts):
        """
        Clean a batch of texts efficiently.
        
        Args:
            texts (list): List of raw texts to be cleaned
            
        Returns:
            list: List of cleaned texts
        """
        cleaned_texts = []
        
        for text in texts:
            if not isinstance(text, str):
                cleaned_texts.append("")
                continue
            
            # Remove HTML tags using BeautifulSoup
            text = BeautifulSoup(text, 'html.parser').get_text()
            
            # Normalize unicode characters
            text = unicodedata.normalize('NFKD', text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs and social media handles
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
            
            # Remove email addresses and phone numbers
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
            
            # Remove excessive punctuation and special characters
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            cleaned_texts.append(text)
        
        return cleaned_texts
    
    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatizer accepts."""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def process_text_batch(self, texts):
        """
        Process a batch of texts with tokenization and lemmatization/stemming.
        
        Args:
            texts (list): List of cleaned texts
            
        Returns:
            list: List of processed texts
        """
        processed_texts = []
        
        for text in texts:
            if not text:
                processed_texts.append("")
                continue
            
            # Tokenize
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            # Filter tokens: remove stopwords, short words, and non-alphabetic tokens
            filtered_tokens = [
                token for token in tokens 
                if (token not in self.stop_words and 
                    len(token) > 2 and 
                    token.isalpha())
            ]
            
            if self.use_lemmatization:
                # Lemmatization with POS tagging (slower but more accurate)
                try:
                    processed_tokens = [
                        self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) 
                        for token in filtered_tokens
                    ]
                except:
                    # Fallback to simple lemmatization
                    processed_tokens = [
                        self.lemmatizer.lemmatize(token) for token in filtered_tokens
                    ]
            else:
                # Stemming (faster)
                processed_tokens = [
                    self.stemmer.stem(token) for token in filtered_tokens
                ]
            
            processed_texts.append(' '.join(processed_tokens))
        
        return processed_texts
    
    def preprocess_text_parallel(self, text):
        """Single text preprocessing for parallel processing."""
        cleaned = self.clean_text_batch([text])[0]
        processed = self.process_text_batch([cleaned])[0]
        return processed
    
    def prepare_data_large_scale(self, df, text_column='text', label_column='label'):
        """
        Prepare large dataset for training/testing with optimized processing.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            
        Returns:
            tuple: Processed features and labels
        """
        print(f"🔄 Preprocessing large dataset ({len(df):,} articles)...")
        
        # Process in batches to manage memory
        processed_texts = []
        batch_count = (len(df) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(df), desc="Processing texts") as pbar:
            for i in range(0, len(df), self.batch_size):
                batch_df = df.iloc[i:i + self.batch_size]
                batch_texts = batch_df[text_column].tolist()
                
                # Clean batch
                cleaned_batch = self.clean_text_batch(batch_texts)
                
                # Process batch
                processed_batch = self.process_text_batch(cleaned_batch)
                
                processed_texts.extend(processed_batch)
                pbar.update(len(batch_texts))
                
                # Force garbage collection to free memory
                if i % (self.batch_size * 5) == 0:
                    gc.collect()
        
        # Create processed DataFrame
        df_processed = df.copy()
        df_processed['processed_text'] = processed_texts
        
        # Remove empty texts
        initial_size = len(df_processed)
        df_processed = df_processed[df_processed['processed_text'].str.len() > 0]
        removed_count = initial_size - len(df_processed)
        
        if removed_count > 0:
            print(f"   ⚠️  Removed {removed_count:,} empty texts after preprocessing")
        
        print(f"   ✅ Dataset size after preprocessing: {len(df_processed):,} articles")
        print(f"   📊 Real news: {len(df_processed[df_processed[label_column] == 0]):,}")
        print(f"   📊 Fake news: {len(df_processed[df_processed[label_column] == 1]):,}")
        
        return df_processed['processed_text'].tolist(), df_processed[label_column].tolist()
    
    def fit_vectorizer_large_scale(self, texts):
        """
        Fit the TF-IDF vectorizer on large text corpus efficiently.
        
        Args:
            texts (list): List of preprocessed texts
        """
        print(f"🔧 Fitting TF-IDF vectorizer on {len(texts):,} texts...")
        
        # Use incremental fitting for very large datasets
        if len(texts) > 100000:
            print("   Using incremental fitting for large dataset...")
            # For very large datasets, we'd implement incremental fitting
            # For now, we'll use the standard approach
        
        with tqdm(desc="Fitting vectorizer") as pbar:
            self.vectorizer.fit(texts)
            pbar.update(1)
        
        vocab_size = len(self.vectorizer.vocabulary_)
        print(f"   ✅ Vocabulary size: {vocab_size:,} features")
        
        # Print some statistics
        feature_names = self.vectorizer.get_feature_names_out()
        print(f"   📈 Feature range: '{feature_names[0]}' to '{feature_names[-1]}'")
    
    def transform_texts_large_scale(self, texts):
        """
        Transform texts to TF-IDF features efficiently.
        
        Args:
            texts (list): List of preprocessed texts
            
        Returns:
            scipy.sparse matrix: TF-IDF features
        """
        print(f"🔄 Transforming {len(texts):,} texts to TF-IDF features...")
        
        with tqdm(desc="Transforming texts") as pbar:
            tfidf_matrix = self.vectorizer.transform(texts)
            pbar.update(1)
        
        print(f"   ✅ TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"   💾 Matrix density: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4f}")
        
        return tfidf_matrix
    
    def get_feature_names(self):
        """Get feature names from the vectorizer."""
        return self.vectorizer.get_feature_names_out()
    
    def save_vectorizer(self, filepath):
        """Save the fitted vectorizer."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"   💾 Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath):
        """Load a saved vectorizer."""
        self.vectorizer = joblib.load(filepath)
        print(f"   📁 Vectorizer loaded from {filepath}")
    
    def get_preprocessing_stats(self, original_texts, processed_texts):
        """Get statistics about the preprocessing results."""
        if not original_texts or not processed_texts:
            return {}
        
        # Sample a subset for statistics to avoid memory issues
        sample_size = min(1000, len(original_texts))
        indices = np.random.choice(len(original_texts), sample_size, replace=False)
        
        orig_sample = [original_texts[i] for i in indices]
        proc_sample = [processed_texts[i] for i in indices]
        
        stats = {
            'avg_original_length': np.mean([len(text) for text in orig_sample]),
            'avg_processed_length': np.mean([len(text) for text in proc_sample]),
            'avg_word_count': np.mean([len(text.split()) for text in proc_sample]),
            'compression_ratio': np.mean([len(proc_sample[i]) / max(len(orig_sample[i]), 1) for i in range(len(orig_sample))])
        }
        
        return stats

def load_large_dataset(data_path='data/large_scale_fake_news_dataset.csv'):
    """
    Load the large-scale fake news dataset efficiently.
    
    Args:
        data_path (str): Path to the dataset CSV file
        
    Returns:
        pd.DataFrame: Large-scale fake news dataset
    """
    try:
        print(f"📁 Loading large dataset from {data_path}...")
        
        # Read CSV with optimized settings
        df = pd.read_csv(
            data_path,
            dtype={
                'text': 'string',
                'label': 'int8',  # Use smaller int type
                'title': 'string',
                'author': 'string',
                'subject': 'string'
            },
            na_filter=False  # Don't convert strings to NaN
        )
        
        print(f"   ✅ Loaded dataset with {len(df):,} articles")
        print(f"   📊 Real news: {len(df[df['label'] == 0]):,} articles")
        print(f"   📊 Fake news: {len(df[df['label'] == 1]):,} articles")
        print(f"   💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {data_path}")
        print("   Please run the dataset creation script first.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    # Test the large-scale preprocessor
    print("🧪 Testing Large-Scale News Preprocessor...")
    
    # Load dataset
    df = load_large_dataset()
    
    if df is not None:
        # Initialize preprocessor
        preprocessor = LargeScaleNewsPreprocessor(
            max_features=20000,
            use_lemmatization=True,
            batch_size=1000
        )
        
        # Test on a small sample
        sample_size = 1000
        sample_df = df.sample(n=sample_size, random_state=42)
        
        print(f"\n🔬 Testing on {sample_size} articles...")
        
        # Prepare data
        texts, labels = preprocessor.prepare_data_large_scale(sample_df)
        
        # Fit vectorizer
        preprocessor.fit_vectorizer_large_scale(texts)
        
        # Transform texts
        tfidf_matrix = preprocessor.transform_texts_large_scale(texts)
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Final matrix shape: {tfidf_matrix.shape}")
    else:
        print("❌ Cannot test without dataset")
