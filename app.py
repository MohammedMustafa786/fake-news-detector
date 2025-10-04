#!/usr/bin/env python3
"""
Fake News Detector Web Interface
A user-friendly Streamlit web app for detecting fake news
"""

import streamlit as st
import pickle
import pandas as pd
import time
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="🕵️ Multilingual Fake News Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Load the trained models (cached for performance)"""
    try:
        with open('models_multilingual/multilingual_preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('models_multilingual/multilingual_best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return preprocessor, model, True
    except Exception as e:
        return None, None, False

def predict_article(text, preprocessor, model):
    """Predict if an article is fake or real"""
    try:
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
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': confidence,
            'real_probability': probabilities[0],
            'fake_probability': probabilities[1],
            'detected_language': language,
            'text_length': len(text)
        }
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence >= 0.8:
        return "🔥"
    elif confidence >= 0.6:
        return "💪"
    else:
        return "⚠️"

def get_prediction_emoji(prediction):
    """Get emoji based on prediction"""
    if prediction == "FAKE":
        return "❌"
    else:
        return "✅"

def main():
    # Header
    st.markdown("""
    # 🕵️ Multilingual Fake News Detector
    ### Detect misinformation in English, Spanish, French, and Hindi
    """)
    
    # Load models
    with st.spinner("Loading AI models..."):
        preprocessor, model, success = load_models()
    
    if not success:
        st.error("""
        ❌ **Failed to load models!**
        
        Please ensure the following files exist:
        - `models_multilingual/multilingual_preprocessor.pkl`
        - `models_multilingual/multilingual_best_model.pkl`
        """)
        return
    
    st.success("✅ AI models loaded successfully!")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("""
        ## 📋 How to Use
        1. **Paste** a news article in the text area
        2. **Click** "Analyze Article" 
        3. **Review** the prediction results
        
        ## 🌍 Supported Languages
        - 🇺🇸 **English**
        - 🇪🇸 **Spanish** 
        - 🇫🇷 **French**
        - 🇮🇳 **Hindi**
        
        ## 🎯 Best Results
        - Use **complete articles** (not just titles)
        - **Longer text** = better accuracy
        - Works best with **news-style content**
        
        ## 📊 Confidence Levels
        - 🔥 **80%+** High Confidence
        - 💪 **60-80%** Medium Confidence  
        - ⚠️ **<60%** Low Confidence
        """)
        
        # Model stats
        st.markdown("---")
        st.markdown("""
        ## 🏆 Model Performance
        - **Overall Accuracy:** 92%+
        - **Training Data:** 59,240 articles
        - **Model Type:** Logistic Regression
        - **Features:** 15,000 TF-IDF features
        """)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        article_text = st.text_area(
            "📰 **Paste News Article Here:**",
            height=400,
            placeholder="Paste the complete news article text here...\n\nFor best results, include the full article content (not just headlines).\n\nExample:\n'Breaking News: Scientists Discover Amazing Health Breakthrough\nNew York - Researchers at Columbia University have announced...[continue with full article]'",
            help="Paste the complete article text for best accuracy. Short titles or summaries may give uncertain results."
        )
        
        # Analyze button
        if st.button("🔍 **Analyze Article**", type="primary"):
            if not article_text.strip():
                st.warning("⚠️ Please enter some text to analyze!")
            elif len(article_text.strip()) < 50:
                st.warning("⚠️ Text seems too short. For better results, please paste the complete article.")
            else:
                with st.spinner("🤖 Analyzing article..."):
                    # Add small delay for better UX
                    time.sleep(1)
                    
                    result = predict_article(article_text, preprocessor, model)
                    
                    if result:
                        # Store result in session state for the results column
                        st.session_state.last_result = result
                        st.session_state.last_text = article_text
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        # Show results if available
        if hasattr(st.session_state, 'last_result'):
            result = st.session_state.last_result
            
            # Main prediction
            pred_emoji = get_prediction_emoji(result['prediction'])
            conf_emoji = get_confidence_color(result['confidence'])
            
            if result['prediction'] == "FAKE":
                st.markdown(f"""
                <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #f44336;">
                    <h2>{pred_emoji} FAKE NEWS DETECTED</h2>
                    <p><strong>Confidence:</strong> {conf_emoji} {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;">
                    <h2>{pred_emoji} APPEARS LEGITIMATE</h2>
                    <p><strong>Confidence:</strong> {conf_emoji} {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed results
            st.markdown("---")
            st.markdown("**📈 Detailed Analysis:**")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric(
                    "✅ Real Probability", 
                    f"{result['real_probability']:.1%}",
                    delta=None
                )
                
            with col_b:
                st.metric(
                    "❌ Fake Probability", 
                    f"{result['fake_probability']:.1%}",
                    delta=None
                )
            
            # Additional info
            st.markdown("**🔍 Technical Details:**")
            st.info(f"""
            **🌍 Detected Language:** {result['detected_language'].title()}
            
            **📝 Text Length:** {result['text_length']:,} characters
            
            **⏰ Analysis Time:** {datetime.now().strftime('%H:%M:%S')}
            """)
            
            # Confidence interpretation
            confidence = result['confidence']
            if confidence >= 0.8:
                st.success("🔥 **High Confidence** - Very reliable prediction")
            elif confidence >= 0.6:
                st.warning("💪 **Medium Confidence** - Generally reliable")
            else:
                st.warning("⚠️ **Low Confidence** - Consider manual review")
        
        else:
            # Default state
            st.info("👆 Enter an article above and click 'Analyze' to see results here.")
    
    # Example articles section
    st.markdown("---")
    st.markdown("## 📝 Try These Example Articles")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        if st.button("🩺 **Test Health Misinformation**"):
            st.session_state.example_text = """
BREAKING: Scientists Discover Miracle Cure That Eliminates All Diseases
Doctors around the world are being forced to hide this incredible discovery! A team of researchers has found that mixing lemon juice with baking soda creates a powerful compound that cures cancer, diabetes, and heart disease in just 48 hours. Big Pharma companies are desperately trying to suppress this information because it threatens their billion-dollar profits. Thousands of people have already been cured using this simple home remedy. The medical establishment doesn't want you to know about this natural solution that costs less than $2 and works better than any prescription medicine.
            """
    
    with example_col2:
        if st.button("📊 **Test Legitimate News**"):
            st.session_state.example_text = """
Federal Reserve Announces New Interest Rate Policy
Washington D.C. - The Federal Reserve announced today a 0.25% increase in the federal funds rate following their two-day policy meeting. Fed Chair Jerome Powell cited ongoing inflation concerns as the primary driver behind the decision. The move was anticipated by most economists, with 15 of 18 analysts surveyed by Bloomberg predicting the increase. Stock markets showed mixed reactions to the news, with banking stocks rising 2.3% while tech stocks declined 1.1%. Powell emphasized that future rate decisions will be data-dependent, focusing particularly on employment figures and consumer price index trends over the coming months.
            """
    
    # Show example if selected
    if hasattr(st.session_state, 'example_text'):
        st.markdown("**📋 Example Article (click in text area to edit):**")
        example_text = st.text_area(
            "Example:",
            value=st.session_state.example_text,
            height=150,
            key="example_area"
        )
        
        if st.button("🔍 **Analyze Example**", key="analyze_example"):
            with st.spinner("Analyzing example article..."):
                time.sleep(1)
                result = predict_article(example_text, preprocessor, model)
                if result:
                    st.session_state.last_result = result
                    st.session_state.last_text = example_text
                    st.experimental_rerun()

if __name__ == "__main__":
    main()