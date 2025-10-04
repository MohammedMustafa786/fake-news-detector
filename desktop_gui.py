#!/usr/bin/env python3
"""
Fake News Detector Desktop GUI
A simple desktop application using tkinter for local use
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pickle
from datetime import datetime
import threading

class FakeNewsDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🕵️ Multilingual Fake News Detector")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load models
        self.preprocessor = None
        self.model = None
        self.load_models()
        
        self.setup_ui()
    
    def load_models(self):
        """Load the trained models"""
        try:
            with open('models_multilingual/multilingual_preprocessor.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            with open('models_multilingual/multilingual_best_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
            return False
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🕵️ Multilingual Fake News Detector",
            font=('Arial', 18, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Detect misinformation in English, Spanish, French, and Hindi",
            font=('Arial', 12),
            foreground='gray'
        )
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Input
        input_frame = ttk.LabelFrame(main_frame, text="📰 Article Input", padding="10")
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        
        # Text input area
        self.text_area = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            width=50,
            height=20,
            font=('Arial', 11)
        )
        self.text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Placeholder text
        placeholder_text = (
            "Paste the complete news article here...\\n\\n"
            "For best results:\\n"
            "• Use complete articles (not just titles)\\n"
            "• Longer text = better accuracy\\n"
            "• Works with English, Spanish, French, Hindi\\n\\n"
            "Example:\\n"
            "'Breaking News: Scientists Discover Health Breakthrough\\n"
            "New York - Researchers at Columbia University announced...'"
        )
        self.text_area.insert('1.0', placeholder_text)
        self.text_area.bind('<FocusIn>', self.clear_placeholder)
        self.is_placeholder = True
        
        # Buttons frame
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        button_frame.columnconfigure(0, weight=1)
        
        # Analyze button
        self.analyze_button = ttk.Button(
            button_frame,
            text="🔍 Analyze Article",
            command=self.analyze_article,
            style='Accent.TButton'
        )
        self.analyze_button.grid(row=0, column=0, pady=(0, 5))
        
        # Clear button
        clear_button = ttk.Button(
            button_frame,
            text="🗑️ Clear Text",
            command=self.clear_text
        )
        clear_button.grid(row=0, column=1, padx=(10, 0), pady=(0, 5))
        
        # Example buttons
        example_frame = ttk.Frame(button_frame)
        example_frame.grid(row=1, column=0, columnspan=2, pady=(5, 0))
        
        example_fake_btn = ttk.Button(
            example_frame,
            text="📝 Load Fake Example",
            command=self.load_fake_example
        )
        example_fake_btn.grid(row=0, column=0, padx=(0, 5))
        
        example_real_btn = ttk.Button(
            example_frame,
            text="📝 Load Real Example", 
            command=self.load_real_example
        )
        example_real_btn.grid(row=0, column=1, padx=(5, 0))
        
        # Right panel - Results
        results_frame = ttk.LabelFrame(main_frame, text="📊 Analysis Results", padding="10")
        results_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Results header
        self.result_header = ttk.Label(
            results_frame,
            text="👆 Enter an article and click 'Analyze' to see results",
            font=('Arial', 12),
            foreground='gray',
            anchor='center'
        )
        self.result_header.grid(row=0, column=0, pady=(0, 10))
        
        # Results display area
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            width=40,
            height=15,
            font=('Arial', 11),
            state='disabled'
        )
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            results_frame,
            mode='indeterminate',
            length=300
        )
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_label = ttk.Label(
            status_frame,
            text="✅ Models loaded successfully - Ready to analyze articles",
            font=('Arial', 10),
            foreground='green'
        )
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Model info
        model_info = ttk.Label(
            status_frame,
            text="🏆 Accuracy: 92%+ | 📊 Training Data: 59,240 articles | 🤖 Logistic Regression",
            font=('Arial', 9),
            foreground='blue'
        )
        model_info.grid(row=1, column=0, sticky=tk.W)
    
    def clear_placeholder(self, event):
        """Clear placeholder text when clicked"""
        if self.is_placeholder:
            self.text_area.delete('1.0', tk.END)
            self.text_area.configure(foreground='black')
            self.is_placeholder = False
    
    def clear_text(self):
        """Clear the text area"""
        self.text_area.delete('1.0', tk.END)
        self.is_placeholder = False
    
    def load_fake_example(self):
        """Load a fake news example"""
        fake_example = """BREAKING: Scientists Discover Miracle Cure That Eliminates All Diseases
Doctors around the world are being forced to hide this incredible discovery! A team of researchers has found that mixing lemon juice with baking soda creates a powerful compound that cures cancer, diabetes, and heart disease in just 48 hours. Big Pharma companies are desperately trying to suppress this information because it threatens their billion-dollar profits. Thousands of people have already been cured using this simple home remedy. The medical establishment doesn't want you to know about this natural solution that costs less than $2 and works better than any prescription medicine."""
        
        self.clear_text()
        self.text_area.insert('1.0', fake_example)
    
    def load_real_example(self):
        """Load a real news example"""
        real_example = """Federal Reserve Announces New Interest Rate Policy
Washington D.C. - The Federal Reserve announced today a 0.25% increase in the federal funds rate following their two-day policy meeting. Fed Chair Jerome Powell cited ongoing inflation concerns as the primary driver behind the decision. The move was anticipated by most economists, with 15 of 18 analysts surveyed by Bloomberg predicting the increase. Stock markets showed mixed reactions to the news, with banking stocks rising 2.3% while tech stocks declined 1.1%. Powell emphasized that future rate decisions will be data-dependent, focusing particularly on employment figures and consumer price index trends over the coming months."""
        
        self.clear_text()
        self.text_area.insert('1.0', real_example)
    
    def show_progress(self):
        """Show progress bar"""
        self.progress.grid(row=2, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        self.progress.start()
    
    def hide_progress(self):
        """Hide progress bar"""
        self.progress.stop()
        self.progress.grid_remove()
    
    def analyze_article(self):
        """Analyze the article in a separate thread"""
        article_text = self.text_area.get('1.0', tk.END).strip()
        
        if not article_text or self.is_placeholder:
            messagebox.showwarning("Warning", "Please enter some text to analyze!")
            return
        
        if len(article_text) < 50:
            messagebox.showwarning("Warning", "Text seems too short. For better results, please paste a complete article.")
            return
        
        # Disable button and show progress
        self.analyze_button.configure(state='disabled')
        self.show_progress()
        self.status_label.configure(text="🤖 Analyzing article...", foreground='orange')
        
        # Run analysis in separate thread
        thread = threading.Thread(target=self.perform_analysis, args=(article_text,))
        thread.daemon = True
        thread.start()
    
    def perform_analysis(self, article_text):
        """Perform the actual analysis"""
        try:
            # Detect language
            language = self.preprocessor.detect_language(article_text)
            
            # Preprocess text
            processed_text = self.preprocessor.preprocess_by_language(article_text, language)
            
            # Vectorize
            processed_features = self.preprocessor.vectorizer.transform([processed_text])
            
            # Predict
            prediction = self.model.predict(processed_features)[0]
            probabilities = self.model.predict_proba(processed_features)[0]
            confidence = max(probabilities)
            
            result = {
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'confidence': confidence,
                'real_probability': probabilities[0],
                'fake_probability': probabilities[1],
                'detected_language': language,
                'text_length': len(article_text)
            }
            
            # Update UI on main thread
            self.root.after(0, self.display_results, result)
            
        except Exception as e:
            self.root.after(0, self.display_error, str(e))
    
    def display_results(self, result):
        """Display analysis results"""
        # Hide progress and enable button
        self.hide_progress()
        self.analyze_button.configure(state='normal')
        
        # Format results
        pred_emoji = "❌" if result['prediction'] == "FAKE" else "✅"
        conf_level = "🔥" if result['confidence'] >= 0.8 else "💪" if result['confidence'] >= 0.6 else "⚠️"
        
        results_text = f"""
{pred_emoji} PREDICTION: {result['prediction']}
{conf_level} CONFIDENCE: {result['confidence']:.1%}

📈 DETAILED ANALYSIS:
• Real Probability: {result['real_probability']:.1%}
• Fake Probability: {result['fake_probability']:.1%}

🔍 TECHNICAL DETAILS:
• Detected Language: {result['detected_language'].title()}
• Text Length: {result['text_length']:,} characters
• Analysis Time: {datetime.now().strftime('%H:%M:%S')}

💭 INTERPRETATION:
"""
        
        if result['prediction'] == "FAKE":
            if result['confidence'] >= 0.8:
                results_text += "🚨 HIGH CONFIDENCE - This article shows strong indicators of misinformation or fake news."
            else:
                results_text += "⚠️ MODERATE CONFIDENCE - This article may contain misinformation. Consider fact-checking."
        else:
            if result['confidence'] >= 0.8:
                results_text += "✅ HIGH CONFIDENCE - This article appears to be legitimate news."
            else:
                results_text += "💭 MODERATE CONFIDENCE - This article appears legitimate but may need manual review."
        
        confidence = result['confidence']
        if confidence < 0.6:
            results_text += "\\n\\n⚠️ LOW CONFIDENCE: The model is uncertain. Consider manual fact-checking."
        
        # Display results
        self.results_text.configure(state='normal')
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', results_text)
        self.results_text.configure(state='disabled')
        
        # Update header and status
        self.result_header.configure(
            text=f"{pred_emoji} Analysis Complete - {result['prediction']}", 
            foreground='red' if result['prediction'] == 'FAKE' else 'green'
        )
        self.status_label.configure(text="✅ Analysis complete", foreground='green')
    
    def display_error(self, error_message):
        """Display error message"""
        self.hide_progress()
        self.analyze_button.configure(state='normal')
        self.status_label.configure(text="❌ Error during analysis", foreground='red')
        
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\\n\\n{error_message}")

def main():
    root = tk.Tk()
    app = FakeNewsDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()