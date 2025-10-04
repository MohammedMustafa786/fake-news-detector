import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FakeNewsClassifier:
    """
    A comprehensive fake news classification system with multiple ML algorithms.
    Supports training, evaluation, and model comparison.
    """
    
    def __init__(self):
        self.models = {
            'naive_bayes': MultinomialNB(),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_single_model(self, model_name, X_train, y_train, X_test=None, y_test=None):
        """
        Train a single model and evaluate its performance.
        
        Args:
            model_name (str): Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_test: Testing features (optional)
            y_test: Testing labels (optional)
            
        Returns:
            dict: Model performance metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.models.keys())}")
        
        print(f"Training {model_name}...")
        
        # Train the model
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        # Store the trained model
        self.trained_models[model_name] = model
        
        # Evaluate the model
        results = {}
        
        # Training accuracy
        train_pred = model.predict(X_train)
        results['train_accuracy'] = accuracy_score(y_train, train_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        # Test accuracy (if test data provided)
        if X_test is not None and y_test is not None:
            test_pred = model.predict(X_test)
            results['test_accuracy'] = accuracy_score(y_test, test_pred)
            results['classification_report'] = classification_report(y_test, test_pred)
            results['confusion_matrix'] = confusion_matrix(y_test, test_pred)
        
        self.model_scores[model_name] = results
        
        print(f"{model_name} training completed!")
        print(f"Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        
        if 'test_accuracy' in results:
            print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train all available models and compare their performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features (optional)
            y_test: Testing labels (optional)
            
        Returns:
            dict: Performance comparison of all models
        """
        print("Training all models...")
        print("=" * 50)
        
        all_results = {}
        
        for model_name in self.models.keys():
            results = self.train_single_model(model_name, X_train, y_train, X_test, y_test)
            all_results[model_name] = results
            print("-" * 30)
        
        # Find the best model based on cross-validation score
        best_cv_score = 0
        for model_name, results in all_results.items():
            if results['cv_mean'] > best_cv_score:
                best_cv_score = results['cv_mean']
                self.best_model_name = model_name
                self.best_model = self.trained_models[model_name]
        
        print(f"\nBest model: {self.best_model_name} (CV Score: {best_cv_score:.4f})")
        
        return all_results
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid=None):
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name (str): Name of the model to tune
            X_train: Training features
            y_train: Training labels
            param_grid (dict): Parameter grid for tuning
            
        Returns:
            dict: Best parameters and score
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        # Default parameter grids
        default_param_grids = {
            'naive_bayes': {'alpha': [0.1, 0.5, 1.0, 2.0]},
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if param_grid is None:
            param_grid = default_param_grids.get(model_name, {})
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update the model with best parameters
        self.trained_models[model_name] = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        print(f"Best parameters: {results['best_params']}")
        print(f"Best CV score: {results['best_score']:.4f}")
        
        return results
    
    def predict(self, X, model_name=None):
        """
        Make predictions using a trained model.
        
        Args:
            X: Features to predict
            model_name (str): Name of model to use (uses best model if None)
            
        Returns:
            numpy.ndarray: Predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model has been trained yet")
            model = self.best_model
            used_model = self.best_model_name
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} has not been trained")
            model = self.trained_models[model_name]
            used_model = model_name
        
        predictions = model.predict(X)
        print(f"Predictions made using {used_model}")
        
        return predictions
    
    def predict_proba(self, X, model_name=None):
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict
            model_name (str): Name of model to use (uses best model if None)
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model has been trained yet")
            model = self.best_model
            used_model = self.best_model_name
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} has not been trained")
            model = self.trained_models[model_name]
            used_model = model_name
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            print(f"Probabilities calculated using {used_model}")
            return probabilities
        else:
            raise ValueError(f"Model {used_model} does not support probability predictions")
    
    def save_model(self, model_name, filepath):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self.trained_models[model_name], filepath)
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name, filepath):
        """
        Load a trained model from disk.
        
        Args:
            model_name (str): Name to assign to the loaded model
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        print(f"Model loaded as {model_name} from {filepath}")
    
    def get_model_comparison(self):
        """
        Get a comparison of all trained models.
        
        Returns:
            pd.DataFrame: Comparison table
        """
        if not self.model_scores:
            raise ValueError("No models have been trained yet")
        
        comparison_data = []
        
        for model_name, scores in self.model_scores.items():
            row = {
                'Model': model_name,
                'Train Accuracy': scores.get('train_accuracy', 0),
                'CV Mean': scores.get('cv_mean', 0),
                'CV Std': scores.get('cv_std', 0),
                'Test Accuracy': scores.get('test_accuracy', 0)
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('CV Mean', ascending=False)
    
    def plot_model_comparison(self, save_path=None):
        """
        Plot model comparison visualization.
        
        Args:
            save_path (str): Path to save the plot (optional)
        """
        if not self.model_scores:
            raise ValueError("No models have been trained yet")
        
        comparison_df = self.get_model_comparison()
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: CV Scores
        plt.subplot(2, 2, 1)
        plt.bar(comparison_df['Model'], comparison_df['CV Mean'])
        plt.title('Cross-Validation Scores')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Plot 2: Train vs Test Accuracy
        if 'Test Accuracy' in comparison_df.columns and comparison_df['Test Accuracy'].sum() > 0:
            plt.subplot(2, 2, 2)
            x = np.arange(len(comparison_df))
            width = 0.35
            
            plt.bar(x - width/2, comparison_df['Train Accuracy'], width, label='Train')
            plt.bar(x + width/2, comparison_df['Test Accuracy'], width, label='Test')
            
            plt.title('Train vs Test Accuracy')
            plt.ylabel('Accuracy')
            plt.xticks(x, comparison_df['Model'], rotation=45)
            plt.legend()
        
        # Plot 3: CV Score with Error Bars
        plt.subplot(2, 2, 3)
        plt.errorbar(comparison_df['Model'], comparison_df['CV Mean'], 
                    yerr=comparison_df['CV Std'], fmt='o', capsize=5)
        plt.title('CV Scores with Standard Deviation')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self):
        """
        Generate a comprehensive model performance report.
        
        Returns:
            dict: Detailed performance report
        """
        if not self.model_scores:
            raise ValueError("No models have been trained yet")
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models_trained': list(self.trained_models.keys()),
            'best_model': self.best_model_name,
            'model_comparison': self.get_model_comparison().to_dict('records'),
            'detailed_scores': self.model_scores
        }
        
        return report

if __name__ == "__main__":
    # Example usage
    from preprocess import NewsPreprocessor, create_sample_data
    from sklearn.model_selection import train_test_split
    
    print("Testing Fake News Classifier...")
    
    # Create sample data
    sample_df = create_sample_data()
    
    # Preprocess data
    preprocessor = NewsPreprocessor(max_features=1000)
    texts, labels = preprocessor.prepare_data(sample_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    # Transform to TF-IDF
    preprocessor.fit_vectorizer(X_train)
    X_train_tfidf = preprocessor.transform_texts(X_train)
    X_test_tfidf = preprocessor.transform_texts(X_test)
    
    # Train classifier
    classifier = FakeNewsClassifier()
    results = classifier.train_all_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Show comparison
    print("\nModel Comparison:")
    print(classifier.get_model_comparison())
    
    # Make predictions
    predictions = classifier.predict(X_test_tfidf)
    probabilities = classifier.predict_proba(X_test_tfidf)
    
    print(f"\nSample predictions: {predictions[:5]}")
    print(f"Sample probabilities: {probabilities[:5]}")