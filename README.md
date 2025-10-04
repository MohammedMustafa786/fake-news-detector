# 🕵️ Multilingual Fake News Detector

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/yourusername/fake-news-detector/deploy.yml?branch=main)](https://github.com/yourusername/fake-news-detector/actions)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/accuracy-92%25%2B-brightgreen)](README.md)
[![Languages](https://img.shields.io/badge/languages-EN%20%7C%20ES%20%7C%20FR%20%7C%20HI-orange)](README.md)

> 🎯 **A production-ready AI system that detects fake news across multiple languages with 92%+ accuracy**

![Demo Screenshot](https://via.placeholder.com/800x400/667eea/ffffff?text=Fake+News+Detector+Demo)

## ✨ **Key Features**

🌍 **Multilingual Support** - English, Spanish, French, Hindi  
🎯 **High Accuracy** - 92%+ detection rate on 59,240+ articles  
⚡ **Real-time Processing** - Sub-second response times  
🖥️ **Multiple Interfaces** - Desktop GUI, REST API, Web App  
🚀 **Production Ready** - Docker, CI/CD, comprehensive testing  
🧠 **Advanced AI** - Logistic Regression with 15K TF-IDF features  

---

## 🚀 **Quick Start**

### **Option 1: Desktop Application**
```bash
# Clone and run
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
pip install -r requirements-deploy.txt
python desktop_gui.py
```

### **Option 2: Web API**
```bash
# Start API server
python api.py
# Open http://localhost:5000
```

### **Option 3: Docker**
```bash
docker build -t fake-news-detector .
docker run -p 5000:5000 fake-news-detector
```

---

## 📊 **Performance Metrics**

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 92.1% |
| **Health Misinformation Detection** | 89.9% confidence |
| **Financial Scam Detection** | 69.7% confidence |
| **Political Conspiracy Detection** | 69.0% confidence |
| **Legitimate News Recognition** | 90%+ confidence |
| **Processing Speed** | < 1 second |

---

## 🛠️ **Available Interfaces**

### **🖥️ Desktop GUI**
- User-friendly tkinter interface
- Real-time analysis with progress indicators
- Built-in example articles
- Detailed confidence scoring

### **🌐 REST API** 
- Professional Flask API with documentation
- JSON responses with detailed metrics
- Health monitoring endpoints
- CORS enabled for web integration

### **📱 Web Application**
- Interactive Streamlit interface
- Live demo capabilities  
- Example article testing
- Professional UI/UX

---

## 🧪 **Testing & Validation**

```bash
# Run comprehensive test suite
python test_project_comprehensive.py

# Test with realistic fake news examples
python realistic_fake_news_test.py

# Test multilingual capabilities
python test_multilingual_fixed.py
```

**Test Results:**
- ✅ All core functionality tests passed
- ✅ Multilingual processing verified
- ✅ API endpoints validated
- ✅ Model loading and prediction confirmed

---

## 🏗️ **Project Architecture**

```
fake-news-detector/
├── 🤖 AI Models
│   ├── models_multilingual/          # Production models
│   └── data/                        # Training datasets (59K+ articles)
├── 🖥️ Interfaces  
│   ├── desktop_gui.py              # Desktop application
│   ├── api.py                      # REST API server
│   └── app.py                      # Streamlit web app
├── 🔧 Core Components
│   ├── multilingual_preprocessing.py
│   ├── train_multilingual_model.py
│   └── test_project_comprehensive.py
└── 📦 Deployment
    ├── Dockerfile                  # Container configuration
    ├── requirements-deploy.txt     # Production dependencies
    └── .github/workflows/          # CI/CD automation
```

---

## 🌐 **API Usage Examples**

### **Python**
```python
import requests

response = requests.post('http://localhost:5000/predict', json={
    'text': 'Your news article text here...'
})

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### **JavaScript**
```javascript
const response = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: articleText })
});

const result = await response.json();
console.log(`Prediction: ${result.prediction}`);
```

### **cURL**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article here..."}'
```

---

## 🎯 **Best Performance On**

✅ **Health Misinformation** - Miracle cures, medical conspiracies  
✅ **Financial Scams** - Get-rich-quick schemes, crypto scams  
✅ **Political Conspiracies** - Election fraud, government plots  
✅ **Clickbait Articles** - Sensational headlines, false claims  
✅ **Multilingual Content** - English, Spanish, French, Hindi  

---

## 🚀 **Deployment Options**

### **Local Development**
```bash
python desktop_gui.py          # Desktop app
python api.py                  # Web API
streamlit run app.py          # Streamlit app
```

### **Production Deployment**
- 🐳 **Docker**: Containerized deployment
- ☁️ **Cloud**: AWS, Google Cloud, Azure compatible
- 🔄 **CI/CD**: GitHub Actions automation
- 📊 **Monitoring**: Health checks and logging

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

---

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Built with**: scikit-learn, NLTK, Flask, Streamlit, tkinter
- **Data Sources**: Verified multilingual news datasets
- **ML Techniques**: TF-IDF vectorization, ensemble methods
- **Inspiration**: Academic research in computational linguistics

---

## 📞 **Support**

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/fake-news-detector/issues)
- 📖 **Documentation**: [Deployment Guide](DEPLOYMENT_GUIDE.md)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/fake-news-detector/discussions)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

**🚀 Ready to detect fake news? [Get Started](#-quick-start)**

</div>