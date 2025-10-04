#!/usr/bin/env python3
"""
Fake News Detector REST API
Flask API for integrating fake news detection into other applications
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pickle
import logging
from datetime import datetime
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
preprocessor = None
model = None
model_loaded = False

def load_models():
    """Load the trained models"""
    global preprocessor, model, model_loaded
    
    try:
        with open('models_multilingual/multilingual_preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('models_multilingual/multilingual_best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        model_loaded = True
        logger.info("Models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        model_loaded = False
        return False

def predict_article(text):
    """Predict if an article is fake or real"""
    try:
        if not model_loaded:
            raise Exception("Models not loaded")
        
        # Detect language
        language = preprocessor.detect_language(text)
        
        # Preprocess text
        processed_text = preprocessor.preprocess_by_language(text, language)
        
        # Vectorize
        processed_features = preprocessor.vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(processed_features)[0]
        probabilities = model.predict_proba(processed_features)[0]
        confidence = max(probabilities)
        
        return {
            'success': True,
            'prediction': 'fake' if prediction == 1 else 'real',
            'confidence': float(confidence),
            'probabilities': {
                'real': float(probabilities[0]),
                'fake': float(probabilities[1])
            },
            'detected_language': language,
            'text_length': len(text),
            'analysis_time': datetime.now().isoformat(),
            'confidence_level': 'high' if confidence >= 0.8 else 'medium' if confidence >= 0.6 else 'low'
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'analysis_time': datetime.now().isoformat()
        }

# API Routes

@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🕵️ Fake News Detector API</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 1000px; 
                margin: 0 auto; 
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 30px; 
                border-radius: 10px; 
                text-align: center;
                margin-bottom: 30px;
            }
            .section { 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .endpoint { 
                background: #f8f9fa; 
                padding: 15px; 
                border-left: 4px solid #007bff; 
                margin: 10px 0;
                border-radius: 0 5px 5px 0;
            }
            .status { 
                display: inline-block; 
                padding: 5px 10px; 
                border-radius: 20px; 
                font-size: 12px;
                font-weight: bold;
            }
            .status.ready { background: #d4edda; color: #155724; }
            .status.error { background: #f8d7da; color: #721c24; }
            code { 
                background: #f1f1f1; 
                padding: 2px 5px; 
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }
            .demo-form {
                background: #fff3cd;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #ffc107;
            }
            textarea {
                width: 100%;
                height: 150px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-family: Arial, sans-serif;
            }
            button {
                background: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover { background: #0056b3; }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }
            .result-fake { background: #f8d7da; border-left: 4px solid #dc3545; }
            .result-real { background: #d4edda; border-left: 4px solid #28a745; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🕵️ Multilingual Fake News Detector API</h1>
            <p>REST API for detecting misinformation in English, Spanish, French, and Hindi</p>
            <span class="status {{ 'ready' if model_status else 'error' }}">
                {{ 'Models Loaded ✅' if model_status else 'Models Error ❌' }}
            </span>
        </div>

        <div class="section">
            <h2>📋 API Documentation</h2>
            
            <div class="endpoint">
                <h3>🔍 POST /predict</h3>
                <p><strong>Description:</strong> Analyze a news article for fake news detection</p>
                <p><strong>Content-Type:</strong> <code>application/json</code></p>
                <p><strong>Request Body:</strong></p>
                <pre><code>{
    "text": "Your news article text here..."
}</code></pre>
                
                <p><strong>Response:</strong></p>
                <pre><code>{
    "success": true,
    "prediction": "fake" | "real",
    "confidence": 0.85,
    "probabilities": {
        "real": 0.15,
        "fake": 0.85
    },
    "detected_language": "english",
    "text_length": 1250,
    "confidence_level": "high" | "medium" | "low",
    "analysis_time": "2024-01-01T12:00:00"
}</code></pre>
            </div>

            <div class="endpoint">
                <h3>📊 GET /health</h3>
                <p><strong>Description:</strong> Check API health and model status</p>
                <p><strong>Response:</strong></p>
                <pre><code>{
    "status": "healthy",
    "models_loaded": true,
    "supported_languages": ["english", "spanish", "french", "hindi"],
    "api_version": "1.0.0"
}</code></pre>
            </div>
        </div>

        <div class="section">
            <h2>🎯 Supported Features</h2>
            <ul>
                <li>🌍 <strong>Multilingual:</strong> English, Spanish, French, Hindi</li>
                <li>📊 <strong>High Accuracy:</strong> 92%+ on test data</li>
                <li>⚡ <strong>Fast Processing:</strong> Sub-second response times</li>
                <li>🔗 <strong>CORS Enabled:</strong> Use from web applications</li>
                <li>📈 <strong>Confidence Scores:</strong> Reliability indicators</li>
            </ul>
        </div>

        <div class="demo-form">
            <h2>🧪 Try the API</h2>
            <p>Test the fake news detector with your own article:</p>
            
            <textarea id="articleText" placeholder="Paste a news article here for analysis...

Example:
BREAKING: Scientists Discover Miracle Cure
Doctors around the world are hiding this incredible discovery! A simple mixture of lemon juice and baking soda can cure all diseases in 48 hours..."></textarea>
            
            <br><br>
            <button onclick="analyzeArticle()">🔍 Analyze Article</button>
            
            <div id="result"></div>
        </div>

        <div class="section">
            <h2>💻 Usage Examples</h2>
            
            <h3>🐍 Python</h3>
            <pre><code>import requests

url = "http://localhost:5000/predict"
data = {"text": "Your news article here..."}

response = requests.post(url, json=data)
result = response.json()

if result['success']:
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
else:
    print(f"Error: {result['error']}")
</code></pre>

            <h3>📜 JavaScript</h3>
            <pre><code>const analyzeArticle = async (articleText) => {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: articleText })
    });
    
    const result = await response.json();
    
    if (result.success) {
        console.log(`Prediction: ${result.prediction}`);
        console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    } else {
        console.error(`Error: ${result.error}`);
    }
};
</code></pre>

            <h3>🌐 cURL</h3>
            <pre><code>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Your news article here..."}'
</code></pre>
        </div>

        <script>
            async function analyzeArticle() {
                const text = document.getElementById('articleText').value;
                const resultDiv = document.getElementById('result');
                
                if (!text.trim()) {
                    alert('Please enter some text to analyze!');
                    return;
                }
                
                resultDiv.innerHTML = '🤖 Analyzing article...';
                resultDiv.style.display = 'block';
                resultDiv.className = '';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        const emoji = result.prediction === 'fake' ? '❌' : '✅';
                        const confidenceEmoji = result.confidence_level === 'high' ? '🔥' : 
                                               result.confidence_level === 'medium' ? '💪' : '⚠️';
                        
                        resultDiv.className = result.prediction === 'fake' ? 'result-fake' : 'result-real';
                        resultDiv.innerHTML = `
                            <h3>${emoji} Prediction: ${result.prediction.toUpperCase()}</h3>
                            <p><strong>${confidenceEmoji} Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                            <p><strong>🌍 Language:</strong> ${result.detected_language}</p>
                            <p><strong>📊 Probabilities:</strong> Real ${(result.probabilities.real * 100).toFixed(1)}%, Fake ${(result.probabilities.fake * 100).toFixed(1)}%</p>
                            <p><strong>📝 Text Length:</strong> ${result.text_length.toLocaleString()} characters</p>
                            <p><strong>⏰ Analysis Time:</strong> ${new Date(result.analysis_time).toLocaleTimeString()}</p>
                        `;
                    } else {
                        resultDiv.className = 'result-fake';
                        resultDiv.innerHTML = `<h3>❌ Error</h3><p>${result.error}</p>`;
                    }
                } catch (error) {
                    resultDiv.className = 'result-fake';
                    resultDiv.innerHTML = `<h3>❌ Network Error</h3><p>${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html_template, model_status=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if models are loaded
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please check server logs.',
                'analysis_time': datetime.now().isoformat()
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Request must contain "text" field with article content',
                'analysis_time': datetime.now().isoformat()
            }), 400
        
        article_text = data['text']
        
        # Validate text length
        if not article_text or len(article_text.strip()) < 10:
            return jsonify({
                'success': False,
                'error': 'Text too short. Please provide a complete article for best results.',
                'analysis_time': datetime.now().isoformat()
            }), 400
        
        # Perform prediction
        result = predict_article(article_text)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'analysis_time': datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'models_loaded': model_loaded,
        'supported_languages': ['english', 'spanish', 'french', 'hindi'],
        'api_version': '1.0.0',
        'server_time': datetime.now().isoformat(),
        'model_info': {
            'type': 'Logistic Regression',
            'accuracy': '92%+',
            'training_data': '59,240 articles',
            'features': '15,000 TF-IDF features'
        } if model_loaded else None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/predict', '/health', '/'],
        'analysis_time': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'analysis_time': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Fake News Detector API...")
    
    # Load models on startup
    if load_models():
        print("✅ Models loaded successfully")
        print("🌍 Supported languages: English, Spanish, French, Hindi")
        print("📊 API ready for requests")
        print()
        print("🔗 API Endpoints:")
        print("   • http://localhost:5000/ (Documentation)")
        print("   • http://localhost:5000/predict (POST)")
        print("   • http://localhost:5000/health (GET)")
        print()
    else:
        print("❌ Failed to load models - API will return errors")
        print("📁 Make sure model files exist in models_multilingual/")
        print()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=False,  # Set to True for development
        threaded=True  # Handle multiple requests
    )