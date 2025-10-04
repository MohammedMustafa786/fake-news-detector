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

class NewsPreprocessor:
    """
    A comprehensive text preprocessing class for fake news detection.
    Handles text cleaning, feature extraction, and data preparation.
    """
    
    def __init__(self, max_features=10000, use_lemmatization=True):
        self.max_features = max_features
        self.use_lemmatization = use_lemmatization
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
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
            ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger'),
            ('corpora/omw-1.4', 'omw-1.4')
        ]
        
        for data_path, download_name in required_data:
            try:
                nltk.data.find(data_path)
            except LookupError:
                print(f"Downloading {download_name}...")
                nltk.download(download_name, quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Advanced text cleaning with comprehensive preprocessing.
        
        Args:
            text (str): Raw text to be cleaned
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags using BeautifulSoup
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and social media handles
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove excessive punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_wordnet_pos(self, word):
        """
        Map POS tag to first character lemmatizer accepts.
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def tokenize_and_process(self, text):
        """
        Advanced tokenization with lemmatization or stemming.
        
        Args:
            text (str): Cleaned text
            
        Returns:
            str: Processed text with lemmatized/stemmed words
        """
        if not text:
            return ""
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens: remove stopwords, short words, and non-alphabetic tokens
        filtered_tokens = [
            token for token in tokens 
            if (token not in self.stop_words and 
                len(token) > 2 and 
                token.isalpha())
        ]
        
        if self.use_lemmatization:
            # Lemmatization with POS tagging
            processed_tokens = [
                self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) 
                for token in filtered_tokens
            ]
        else:
            # Stemming
            processed_tokens = [
                self.stemmer.stem(token) for token in filtered_tokens
            ]
        
        return ' '.join(processed_tokens)
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Fully preprocessed text
        """
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_process(cleaned)
        return processed
    
    def prepare_data(self, df, text_column='text', label_column='label'):
        """
        Prepare dataset for training/testing.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            
        Returns:
            tuple: Processed features and labels
        """
        print("Preprocessing text data...")
        
        # Preprocess text
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        print(f"Dataset size after preprocessing: {len(df)}")
        
        return df['processed_text'].tolist(), df[label_column].tolist()
    
    def fit_vectorizer(self, texts):
        """
        Fit the TF-IDF vectorizer on training texts.
        
        Args:
            texts (list): List of preprocessed texts
        """
        print("Fitting TF-IDF vectorizer...")
        self.vectorizer.fit(texts)
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def transform_texts(self, texts):
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts (list): List of preprocessed texts
            
        Returns:
            scipy.sparse matrix: TF-IDF features
        """
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """
        Get feature names from the vectorizer.
        
        Returns:
            list: Feature names
        """
        return self.vectorizer.get_feature_names_out()
    
    def save_vectorizer(self, filepath):
        """
        Save the fitted vectorizer.
        
        Args:
            filepath (str): Path to save the vectorizer
        """
        joblib.dump(self.vectorizer, filepath)
        print(f"Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath):
        """
        Load a saved vectorizer.
        
        Args:
            filepath (str): Path to the saved vectorizer
        """
        self.vectorizer = joblib.load(filepath)
        print(f"Vectorizer loaded from {filepath}")

def load_comprehensive_dataset(data_path='data/comprehensive_fake_news_dataset.csv'):
    """
    Load the comprehensive fake news dataset.
    
    Args:
        data_path (str): Path to the dataset CSV file
        
    Returns:
        pd.DataFrame: Comprehensive fake news dataset
    """
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded comprehensive dataset with {len(df)} articles")
        print(f"- Real news: {len(df[df['label'] == 0])} articles")
        print(f"- Fake news: {len(df[df['label'] == 1])} articles")
        return df
    except FileNotFoundError:
        print(f"Dataset file not found: {data_path}")
        print("Creating sample dataset instead...")
        return create_sample_data()

def create_sample_data():
    """
    Create a sample dataset for demonstration purposes.
    
    Returns:
        pd.DataFrame: Sample fake news dataset
    """
    fake_news = [
        "BREAKING: Scientists discover aliens living among us, government tries to cover up!",
        "You won't believe this miracle cure that doctors don't want you to know about!",
        "SHOCKING: Celebrity caught in massive scandal, career over!",
        "This one weird trick will make you rich overnight, banks hate it!",
        "URGENT: New study proves vaccines cause autism, mainstream media silent!",
        "Local man discovers secret to eternal youth, pharmaceutical companies furious!",
        "EXPOSED: Government mind control through 5G towers, wake up people!",
        "Celebrity endorses this amazing weight loss pill, lose 50 pounds in a week!",
        "BREAKING: Evidence of voter fraud discovered, election was rigged!",
        "This natural remedy cures cancer, but Big Pharma keeps it secret!"
    ]
    
    real_news = [
        "The Federal Reserve announced a 0.25% interest rate increase following today's meeting.",
        "New archaeological findings in Egypt reveal insights into ancient Egyptian daily life.",
        "Climate scientists report record-breaking temperatures recorded across multiple regions this summer.",
        "The stock market closed mixed today with technology sectors showing modest gains.",
        "Local university receives grant funding for renewable energy research project.",
        "City council approves new infrastructure improvements for downtown area.",
        "Healthcare officials recommend updated vaccination schedules for the upcoming season.",
        "International trade negotiations continue between major economic partners.",
        "Space agency successfully launches satellite for weather monitoring purposes.",
        "Educational reforms aim to improve student outcomes in underperforming districts."
    ]
    
    # Create DataFrame
    data = []
    
    # Add fake news (label = 1)
    for text in fake_news:
        data.append({'text': text, 'label': 1})
    
    # Add real news (label = 0)
    for text in real_news:
        data.append({'text': text, 'label': 0})
    
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Example usage
    print("Creating sample dataset...")
    sample_df = create_sample_data()
    print(f"Sample dataset created with {len(sample_df)} articles")
    print("\nFirst few rows:")
    print(sample_df.head())
    
    # Initialize preprocessor
    preprocessor = NewsPreprocessor(max_features=5000)
    
    # Prepare data
    texts, labels = preprocessor.prepare_data(sample_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    # Fit vectorizer and transform
    preprocessor.fit_vectorizer(X_train)
    X_train_tfidf = preprocessor.transform_texts(X_train)
    X_test_tfidf = preprocessor.transform_texts(X_test)
    
    print(f"\nTraining set shape: {X_train_tfidf.shape}")
    print(f"Testing set shape: {X_test_tfidf.shape}")