# Fake News Detector 🕵️‍♂️

A comprehensive machine learning system for detecting fake news using natural language processing and multiple classification algorithms.

## 🌟 Features

- **Multiple ML Models**: Naive Bayes, SVM, Random Forest, and Logistic Regression
- **Advanced Text Processing**: TF-IDF vectorization, stemming, and stop word removal
- **Interactive CLI**: Command-line interface for real-time predictions
- **Batch Processing**: Analyze multiple articles from CSV files
- **Model Persistence**: Save and load trained models
- **Comprehensive Evaluation**: Cross-validation, confusion matrices, and performance metrics
- **Easy Setup**: One-command installation and execution

## 🚀 Quick Start

### Installation

1. **Clone or download** the project to your local machine
2. **Navigate to the project directory**:
   ```bash
   cd fake-news-detector
   ```
3. **Activate the virtual environment**:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```
4. **Install additional dependencies** (if needed):
   ```bash
   pip install matplotlib seaborn jupyter
   ```

### Quick Demo

Run the demonstration mode to see the system in action:

```bash
python main.py --demo
```

## 📋 Usage Examples

### 1. Train a Model

**Train with sample data:**
```bash
python main.py --train
```

**Train with custom data:**
```bash
python main.py --train --data data/your_dataset.csv
```

### 2. Make Predictions

**Single text prediction:**
```bash
python main.py --predict "Breaking news: Scientists discover amazing new technology!"
```

**Interactive mode:**
```bash
python main.py --interactive
```

**Batch prediction from file:**
```bash
python main.py --batch data/test_articles.csv results.csv
```

### 3. Python API Usage

```python
from src.predict import quick_predict

# Quick prediction
result = quick_predict("Your news text here")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

```python
from src.preprocess import NewsPreprocessor
from src.model import FakeNewsClassifier
from src.predict import FakeNewsPredictor

# Complete workflow
preprocessor = NewsPreprocessor()
classifier = FakeNewsClassifier()
predictor = FakeNewsPredictor()

# Train your own model
df = pd.read_csv('your_data.csv')
texts, labels = preprocessor.prepare_data(df)
# ... training code ...
```

## 📊 Model Performance

The system trains and compares multiple machine learning models:

- **Naive Bayes**: Fast training, good baseline performance
- **Support Vector Machine**: High accuracy with proper tuning
- **Random Forest**: Robust ensemble method
- **Logistic Regression**: Interpretable linear model

### Performance Metrics

- Cross-validation accuracy
- Precision, recall, and F1-score
- Confusion matrices
- Training and test accuracy comparison

## 🏗️ Project Structure

```
fake-news-detector/
│
├── src/                    # Source code modules
│   ├── preprocess.py       # Text preprocessing and feature extraction
│   ├── model.py           # Machine learning models and training
│   └── predict.py         # Prediction system and utilities
│
├── data/                  # Dataset storage
│   └── sample_dataset.csv # Sample training data
│
├── notebooks/             # Jupyter notebooks for analysis
│   └── analysis.ipynb     # Data exploration and model analysis
│
├── models/                # Saved models (created after training)
│   ├── best_model_*.pkl   # Best performing model
│   ├── vectorizer.pkl     # TF-IDF vectorizer
│   └── *_model.pkl        # Individual model files
│
├── venv/                  # Virtual environment
├── main.py               # Main application script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

### Preprocessing Options

```python
preprocessor = NewsPreprocessor(
    max_features=10000,  # Maximum number of TF-IDF features
    ngram_range=(1, 2),  # Use unigrams and bigrams
    min_df=2,           # Minimum document frequency
    max_df=0.95         # Maximum document frequency
)
```

### Model Parameters

Each model can be tuned with custom parameters:

```python
classifier = FakeNewsClassifier()
# Hyperparameter tuning
results = classifier.hyperparameter_tuning('svm', X_train, y_train)
```

## 📈 Data Format

The system expects CSV files with the following format:

```csv
text,label
"Your news article text here",0
"Another article text",1
```

- **text**: The news article content
- **label**: 0 for real news, 1 for fake news

## 🧪 Testing the System

### Quick Test with Sample Data

```bash
# Test preprocessing
python -c "from src.preprocess import create_sample_data; df = create_sample_data(); print(df.head())"

# Test model training
python -c "from src.model import FakeNewsClassifier; print('Model classes:', FakeNewsClassifier().models.keys())"

# Test prediction
python main.py --predict "This is a test news article"
```

### Run All Components

```bash
# Full workflow test
python main.py --train --data data/sample_dataset.csv
python main.py --batch data/sample_dataset.csv test_results.csv
```

## 🎯 Advanced Features

### 1. Model Comparison

```python
classifier = FakeNewsClassifier()
results = classifier.train_all_models(X_train, y_train, X_test, y_test)
comparison = classifier.get_model_comparison()
print(comparison)
```

### 2. Prediction Probabilities

```python
predictor = FakeNewsPredictor(model_path, vectorizer_path)
result = predictor.predict_single(text, return_probability=True)
print(f"Fake probability: {result['probability']['fake']:.2%}")
```

### 3. Batch Analysis

```python
predictor = FakeNewsPredictor()
predictions = predictor.predict_batch(texts)
summary = predictor.get_prediction_summary(predictions)
print(f"Fake news percentage: {summary['fake_news_percentage']:.1f}%")
```

## 🔍 Understanding the Results

### Prediction Output

```json
{
    "text": "Your input text",
    "prediction": 1,
    "label": "FAKE NEWS",
    "confidence": 0.85,
    "probability": {
        "real": 0.15,
        "fake": 0.85
    },
    "model_used": "naive_bayes",
    "timestamp": "2024-01-01 12:00:00"
}
```

### Interpretation Guide

- **Label**: Final classification (REAL NEWS or FAKE NEWS)
- **Confidence**: Maximum probability (higher = more certain)
- **Probability**: Individual class probabilities
- **Model Used**: Which algorithm made the prediction

## 📚 Technical Details

### Text Preprocessing Pipeline

1. **Text Cleaning**: Remove URLs, emails, HTML tags, special characters
2. **Tokenization**: Split text into individual words
3. **Stop Word Removal**: Remove common words (the, and, is, etc.)
4. **Stemming**: Reduce words to root form (running → run)
5. **TF-IDF Vectorization**: Convert text to numerical features

### Machine Learning Pipeline

1. **Data Splitting**: Train/test split with stratification
2. **Feature Engineering**: TF-IDF with n-grams
3. **Model Training**: Multiple algorithms in parallel
4. **Cross-Validation**: 5-fold CV for robust evaluation
5. **Model Selection**: Best model based on CV score
6. **Final Evaluation**: Performance on held-out test set

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Ensure you're in the correct directory and virtual environment is activated
cd fake-news-detector
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

**2. NLTK Data Missing**
```python
# The system automatically downloads required NLTK data
# If issues persist, manually download:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**3. Memory Issues with Large Datasets**
```python
# Reduce feature count
preprocessor = NewsPreprocessor(max_features=1000)
```

**4. Model Files Not Found**
```bash
# Retrain models
python main.py --train
```

## 🤝 Contributing

Contributions are welcome! Here are some ways to contribute:

1. **Add new models**: Implement additional ML algorithms
2. **Improve preprocessing**: Add more text cleaning techniques
3. **Enhance evaluation**: Add more metrics and visualizations
4. **Optimize performance**: Improve speed and memory usage
5. **Add features**: Interactive web interface, API endpoints

## 📄 License

This project is open source and available under the MIT License.

## 🔗 References

- **Scikit-learn**: Machine learning library
- **NLTK**: Natural language processing toolkit
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the example usage
3. Test with the demo mode first
4. Ensure all dependencies are installed correctly

---

**Happy fake news detecting! 🎯**