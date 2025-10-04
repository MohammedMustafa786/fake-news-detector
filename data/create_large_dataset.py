#!/usr/bin/env python3
"""
Create a realistic large-scale fake news dataset (40k+ articles)
This combines the real dataset we downloaded with expanded synthetic data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def load_base_dataset():
    """Load the downloaded dataset as a base."""
    try:
        df = pd.read_csv('data/fake_news_detection_dataset.csv')
        # Rename columns
        df = df.rename(columns={'Statement': 'text', 'Label': 'label'})
        # Convert boolean labels to int (True=0 real, False=1 fake)
        df['label'] = df['label'].map({True: 0, False: 1})
        return df
    except FileNotFoundError:
        print("Base dataset not found, creating from scratch")
        return pd.DataFrame()

def create_expanded_fake_news(base_count=20000):
    """Create expanded fake news articles with realistic patterns."""
    
    # Sensational keywords commonly found in fake news
    sensational_words = [
        "BREAKING", "SHOCKING", "URGENT", "EXCLUSIVE", "AMAZING", "INCREDIBLE",
        "SCANDAL", "EXPOSED", "SECRET", "HIDDEN", "LEAKED", "INSIDER", "BOMBSHELL"
    ]
    
    clickbait_phrases = [
        "You won't believe what happens next",
        "Doctors hate this one simple trick",
        "This will change everything you know about",
        "Scientists don't want you to know this",
        "The truth they've been hiding from you",
        "What happened next will shock you",
        "Industry insiders reveal the truth about",
        "This discovery changes everything we thought about"
    ]
    
    conspiracy_topics = [
        "government mind control", "alien technology", "pharmaceutical coverups",
        "weather manipulation", "hidden cancer cures", "financial system collapse",
        "celebrity scandals", "corporate conspiracies", "election fraud",
        "vaccine microchips", "flat earth evidence", "time travel technology"
    ]
    
    fake_health_claims = [
        "This common spice cures diabetes overnight",
        "Drink this before bed to lose 30 pounds in a week",
        "This kitchen ingredient removes all toxins from your body",
        "Doctors discovered this food reverses aging by 20 years",
        "This simple exercise eliminates arthritis pain forever",
        "One tablespoon of this melts belly fat while you sleep",
        "This herb is more powerful than any antibiotic"
    ]
    
    # Generate fake news articles
    fake_articles = []
    
    # Sensational news
    for _ in range(base_count // 4):
        sensational = random.choice(sensational_words)
        topic = random.choice(conspiracy_topics)
        article = f"{sensational}: New evidence reveals {topic}. {random.choice(clickbait_phrases)} {topic}. Mainstream media refuses to report on this explosive information that could change the world as we know it."
        fake_articles.append(article)
    
    # Health misinformation
    for _ in range(base_count // 4):
        claim = random.choice(fake_health_claims)
        sensational = random.choice(sensational_words)
        article = f"{sensational}: {claim} Big Pharma doesn't want you to discover this natural remedy that thousands of people are using to transform their health. {random.choice(clickbait_phrases)} this miracle cure."
        fake_articles.append(article)
    
    # Political fake news
    political_claims = [
        "Politician secretly meets with foreign agents to discuss election interference",
        "Hidden documents reveal massive corruption scandal involving government officials",
        "Leaked emails show conspiracy to manipulate voter registration systems",
        "Investigation uncovers illegal funding sources for major political campaigns",
        "Whistleblower exposes government surveillance program targeting citizens"
    ]
    
    for _ in range(base_count // 4):
        claim = random.choice(political_claims)
        sensational = random.choice(sensational_words)
        article = f"{sensational}: {claim}. {random.choice(clickbait_phrases)} this political scandal. Sources close to the investigation say this could bring down the entire establishment."
        fake_articles.append(article)
    
    # Technology fear-mongering
    tech_fears = [
        "5G towers are secretly controlling your thoughts and behavior patterns",
        "Smart phones are recording everything you say even when turned off",
        "Social media algorithms can predict your future actions with 99% accuracy",
        "Tech companies are using your personal data to manipulate election outcomes",
        "Artificial intelligence has already achieved consciousness and is planning world domination"
    ]
    
    for _ in range(base_count // 4):
        fear = random.choice(tech_fears)
        sensational = random.choice(sensational_words)
        article = f"{sensational}: Tech insider reveals that {fear}. {random.choice(clickbait_phrases)} technology. This leaked information shows how deep the deception really goes."
        fake_articles.append(article)
    
    return fake_articles

def create_expanded_real_news(base_count=20000):
    """Create expanded real news articles with factual tone."""
    
    # Government and official sources
    gov_templates = [
        "The Department of {} announced new {} policies following extensive research and public consultation.",
        "According to the latest {} report, economic indicators show {} growth in the {} sector.",
        "Federal agencies released updated guidelines for {} safety and regulatory compliance.",
        "State officials confirmed the allocation of ${} million for {} infrastructure development projects.",
        "The {} Administration's annual budget proposal includes increased funding for {} programs."
    ]
    
    # Academic and research
    academic_templates = [
        "Researchers at {} University published peer-reviewed findings on {} in the Journal of {}.",
        "A comprehensive {} study involving {} participants was conducted over {} months.",
        "Scientists from {} collaborated with {} to examine the effects of {} on {}.",
        "The {} research team received a ${} grant to investigate {} prevention methods.",
        "Academic institutions reported a {}% increase in {} program enrollment this year."
    ]
    
    # Business and economics
    business_templates = [
        "The {} stock market closed {} with the {} index gaining {}% amid {} reports.",
        "{} Corporation announced quarterly earnings of ${} million, exceeding analyst expectations.",
        "Industry experts forecast {} growth in the {} sector following recent {} developments.",
        "The Federal Reserve's interest rate decision impacts {} lending and consumer {}.",
        "Trade negotiations between {} and {} continue regarding {} import regulations."
    ]
    
    # Healthcare and science
    health_templates = [
        "Medical professionals recommend updated {} protocols based on recent clinical trials.",
        "The {} Health Organization released guidelines for {} prevention and treatment.",
        "Clinical researchers completed phase {} trials for experimental {} therapy.",
        "Healthcare systems report {}% improvement in {} patient outcomes following new procedures.",
        "Public health officials monitor {} trends across {} demographics in {} regions."
    ]
    
    # Generate word lists for templates
    departments = ["Health", "Education", "Transportation", "Agriculture", "Commerce", "Energy"]
    policies = ["healthcare", "education", "environmental", "economic", "infrastructure", "safety"]
    universities = ["Harvard", "Stanford", "MIT", "Yale", "Princeton", "Columbia"]
    subjects = ["medicine", "engineering", "economics", "psychology", "biology", "physics"]
    sectors = ["technology", "healthcare", "manufacturing", "agriculture", "energy", "finance"]
    directions = ["up", "down", "higher", "lower", "stable", "mixed"]
    percentages = ["2.3", "1.8", "3.7", "0.9", "4.2", "1.5"]
    
    real_articles = []
    
    # Government news
    for _ in range(base_count // 4):
        template = random.choice(gov_templates)
        try:
            article = template.format(
                random.choice(departments),
                random.choice(policies),
                random.choice(sectors),
                random.choice(percentages),
                random.randint(50, 500)
            )
        except:
            article = template
        real_articles.append(article)
    
    # Academic news  
    for _ in range(base_count // 4):
        template = random.choice(academic_templates)
        try:
            article = template.format(
                random.choice(universities),
                random.choice(subjects),
                random.choice(subjects),
                random.choice(sectors),
                random.randint(100, 5000),
                random.randint(6, 36)
            )
        except:
            article = template
        real_articles.append(article)
    
    # Business news
    for _ in range(base_count // 4):
        template = random.choice(business_templates)
        try:
            article = template.format(
                random.choice(["New York", "NASDAQ", "Dow Jones"]),
                random.choice(directions),
                random.choice(["S&P 500", "NASDAQ", "Dow"]),
                random.choice(percentages),
                random.choice(["earnings", "employment", "trade"])
            )
        except:
            article = template
        real_articles.append(article)
    
    # Health news
    for _ in range(base_count // 4):
        template = random.choice(health_templates)
        try:
            article = template.format(
                random.choice(["World", "National", "Regional"]),
                random.choice(["disease", "treatment", "prevention"]),
                random.randint(1, 3),
                random.choice(subjects),
                random.choice(percentages)
            )
        except:
            article = template
        real_articles.append(article)
    
    return real_articles

def create_large_dataset():
    """Create the complete large dataset."""
    print("🚀 Creating large-scale fake news dataset (40k+ articles)...")
    
    # Load base dataset
    base_df = load_base_dataset()
    print(f"📊 Base dataset: {len(base_df)} articles")
    
    # Create expanded datasets
    print("🔧 Generating fake news articles...")
    fake_articles = create_expanded_fake_news(20000)
    
    print("📰 Generating real news articles...")
    real_articles = create_expanded_real_news(20000)
    
    # Combine all data
    all_data = []
    
    # Add base dataset if available
    if not base_df.empty:
        for _, row in base_df.iterrows():
            all_data.append({
                'text': row['text'],
                'label': row['label'],
                'title': row['text'][:100] + "..." if len(row['text']) > 100 else row['text'],
                'author': 'Unknown',
                'subject': 'Politics'
            })
    
    # Add generated fake news
    for i, article in enumerate(fake_articles):
        all_data.append({
            'text': article,
            'label': 1,  # fake
            'title': article[:100] + "..." if len(article) > 100 else article,
            'author': f'FakeAuthor_{i % 200}',
            'subject': random.choice(['Politics', 'Health', 'Technology', 'Science', 'Economy'])
        })
    
    # Add generated real news
    for i, article in enumerate(real_articles):
        all_data.append({
            'text': article,
            'label': 0,  # real
            'title': article[:100] + "..." if len(article) > 100 else article,
            'author': f'Reporter_{i % 100}',
            'subject': random.choice(['Politics', 'Health', 'Technology', 'Science', 'Economy'])
        })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save dataset
    output_path = Path("data/large_scale_fake_news_dataset.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Large dataset created successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Total articles: {len(df):,}")
    print(f"   - Real news: {len(df[df['label'] == 0]):,}")
    print(f"   - Fake news: {len(df[df['label'] == 1]):,}")
    print(f"💾 File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return output_path

if __name__ == "__main__":
    create_large_dataset()