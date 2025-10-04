#!/usr/bin/env python3
"""
Script to download and process large-scale Kaggle Fake News Dataset
This dataset contains 40k+ articles for serious machine learning training
"""
import pandas as pd
import numpy as np
import requests
import os
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm

class KaggleDatasetDownloader:
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
    def download_fake_news_dataset(self):
        """Download the large-scale fake news dataset from various sources."""
        print("🔍 Searching for large-scale fake news datasets...")
        
        # Try multiple dataset sources
        datasets = [
            {
                "name": "WELFake Dataset",
                "url": "https://raw.githubusercontent.com/KaiDMML/WELFake_Dataset/master/WELFake_Dataset.csv",
                "size": "72k articles"
            },
            {
                "name": "Fake News Detection Dataset",
                "url": "https://raw.githubusercontent.com/nishitpatel01/Fake_News_Detection/master/train.csv",
                "size": "20k articles"
            }
        ]
        
        downloaded_files = []
        
        for dataset in datasets:
            try:
                print(f"\n📥 Attempting to download: {dataset['name']} ({dataset['size']})")
                filename = self.data_dir / f"{dataset['name'].lower().replace(' ', '_')}.csv"
                
                # Download with progress bar
                response = requests.get(dataset['url'], stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                
                with open(filename, 'wb') as file, tqdm(
                    desc=f"Downloading {dataset['name']}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        size = file.write(chunk)
                        pbar.update(size)
                
                # Verify download
                if filename.exists() and filename.stat().st_size > 1000:
                    print(f"✅ Successfully downloaded: {filename}")
                    downloaded_files.append(filename)
                else:
                    print(f"❌ Failed to download: {dataset['name']}")
                    
            except Exception as e:
                print(f"❌ Error downloading {dataset['name']}: {e}")
                continue
        
        if not downloaded_files:
            print("\n🚨 No datasets downloaded successfully. Creating large synthetic dataset...")
            return self.create_large_synthetic_dataset()
        
        return downloaded_files
    
    def create_large_synthetic_dataset(self):
        """Create a large synthetic dataset if downloads fail."""
        print("\n🏗️ Creating large synthetic dataset (40k articles)...")
        
        # Base templates for fake news
        fake_templates = [
            "BREAKING: {} discovered that will change everything, but {} don't want you to know!",
            "SHOCKING: New study reveals {} causes {}, mainstream media covers it up!",
            "URGENT: {} found to contain dangerous {} that government hides from public!",
            "EXCLUSIVE: Secret {} method that {} use to control the masses exposed!",
            "AMAZING: Local {} discovers simple trick to {}, experts hate this!",
            "WARNING: Your {} is slowly killing you with hidden {}, doctor reveals truth!",
            "SCANDAL: Major {} company caught putting {} in everyday products!",
            "BREAKTHROUGH: Ancient {} secret that modern {} can't explain finally revealed!",
            "ALERT: New {} law designed to {} citizens, insider leaks documents!",
            "MIRACLE: {} grandmother's {} recipe cures {} in days, doctors baffled!",
        ]
        
        # Base templates for real news
        real_templates = [
            "The {} announced {} following today's board meeting and stakeholder consultation.",
            "According to the latest {} report, {} showed a {} increase over the previous quarter.",
            "Researchers at {} published findings on {} in the journal of {}.",
            "The {} department issued new guidelines for {} compliance and implementation.",
            "Local {} received federal funding for {} development and community outreach programs.",
            "The {} committee approved budget allocations for {} infrastructure improvements.",
            "Scientists from {} collaborated on a study examining {} effects on {}.",
            "The {} agency released annual statistics showing {} trends across multiple regions.",
            "Officials announced the completion of {} project after {} months of development.",
            "The {} organization launched an initiative to promote {} in underserved communities.",
        ]
        
        # Generate word lists for templates
        subjects = ["scientists", "doctors", "government", "experts", "researchers", "officials", 
                   "authorities", "specialists", "investigators", "analysts", "corporations"]
        
        objects = ["technology", "medicine", "chemicals", "procedures", "methods", "treatments",
                  "policies", "regulations", "systems", "programs", "initiatives", "studies"]
        
        institutions = ["university", "hospital", "department", "agency", "organization", 
                       "institute", "commission", "bureau", "council", "foundation"]
        
        effects = ["health problems", "financial loss", "privacy violations", "environmental damage",
                  "social issues", "security risks", "economic impact", "public safety concerns"]
        
        # Generate large dataset
        data = []
        total_articles = 40000
        
        print(f"Generating {total_articles} articles...")
        
        with tqdm(total=total_articles, desc="Creating articles") as pbar:
            for i in range(total_articles):
                if i < total_articles // 2:
                    # Generate fake news
                    template = np.random.choice(fake_templates)
                    try:
                        text = template.format(
                            np.random.choice(objects),
                            np.random.choice(subjects),
                            np.random.choice(effects),
                            np.random.choice(institutions)
                        )
                    except:
                        text = template
                    
                    data.append({
                        'text': text,
                        'label': 1,  # fake
                        'title': text[:100] + "..." if len(text) > 100 else text,
                        'author': f"FakeAuthor_{i % 100}",
                        'subject': np.random.choice(['politics', 'health', 'science', 'technology', 'economy'])
                    })
                else:
                    # Generate real news
                    template = np.random.choice(real_templates)
                    try:
                        text = template.format(
                            np.random.choice(institutions),
                            np.random.choice(objects),
                            np.random.choice(['significant', 'notable', 'substantial', 'moderate']),
                            np.random.choice(subjects)
                        )
                    except:
                        text = template
                    
                    data.append({
                        'text': text,
                        'label': 0,  # real
                        'title': text[:100] + "..." if len(text) > 100 else text,
                        'author': f"Reporter_{i % 50}",
                        'subject': np.random.choice(['politics', 'health', 'science', 'technology', 'economy'])
                    })
                
                pbar.update(1)
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save dataset
        filename = self.data_dir / "large_fake_news_dataset.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n✅ Created large dataset: {filename}")
        print(f"   - Total articles: {len(df):,}")
        print(f"   - Real articles: {len(df[df['label'] == 0]):,}")
        print(f"   - Fake articles: {len(df[df['label'] == 1]):,}")
        print(f"   - File size: {filename.stat().st_size / 1024 / 1024:.1f} MB")
        
        return [filename]
    
    def process_downloaded_dataset(self, filepath):
        """Process and standardize downloaded dataset."""
        print(f"\n🔄 Processing dataset: {filepath}")
        
        try:
            # Load dataset
            df = pd.read_csv(filepath)
            print(f"   - Loaded {len(df):,} rows")
            
            # Standardize column names
            column_mapping = {
                'Statement': 'text',
                'Text': 'text',
                'news': 'text', 
                'content': 'text',
                'article': 'text',
                'Label': 'label',
                'class': 'label',
                'target': 'label',
                'fake': 'label',
                'Title': 'title',
                'headline': 'title',
                'Subject': 'subject',
                'topic': 'subject',
                'Author': 'author',
                'writer': 'author'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            if 'text' not in df.columns:
                if 'title' in df.columns:
                    df['text'] = df['title']
                else:
                    print("❌ No text column found in dataset")
                    return None
            
            if 'label' not in df.columns:
                print("❌ No label column found in dataset")
                return None
            
            # Standardize labels (0 = real, 1 = fake)
            if df['label'].dtype == 'object' or df['label'].dtype == 'bool':
                label_mapping = {
                    # True = real news, False = fake news (common in fact-checking datasets)
                    'REAL': 0, 'real': 0, 'Real': 0, 'true': 0, 'True': 0, 'TRUE': 0, True: 0,
                    'FAKE': 1, 'fake': 1, 'Fake': 1, 'false': 1, 'False': 1, 'FALSE': 1, False: 1
                }
                df['label'] = df['label'].map(label_mapping).fillna(df['label'])
            
            # Convert to numeric
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            
            # Remove rows with invalid labels
            df = df.dropna(subset=['label'])
            df = df[df['label'].isin([0, 1])]
            
            # Remove empty texts
            df = df.dropna(subset=['text'])
            df = df[df['text'].str.len() > 10]
            
            # Add missing columns
            if 'title' not in df.columns:
                df['title'] = df['text'].str[:100] + "..."
            
            if 'author' not in df.columns:
                df['author'] = 'Unknown'
            
            if 'subject' not in df.columns:
                df['subject'] = 'News'
            
            print(f"   - Processed {len(df):,} valid articles")
            print(f"   - Real articles: {len(df[df['label'] == 0]):,}")
            print(f"   - Fake articles: {len(df[df['label'] == 1]):,}")
            
            # Save processed dataset
            output_path = filepath.parent / f"processed_{filepath.name}"
            df.to_csv(output_path, index=False)
            print(f"   - Saved processed dataset: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error processing dataset: {e}")
            return None
    
    def run(self):
        """Main execution function."""
        print("🚀 Starting large-scale dataset preparation...")
        
        # Download datasets
        downloaded_files = self.download_fake_news_dataset()
        
        if not downloaded_files:
            print("❌ No datasets available")
            return None
        
        # Process the largest dataset
        largest_file = None
        largest_size = 0
        
        for filepath in downloaded_files:
            if filepath.exists():
                size = filepath.stat().st_size
                if size > largest_size:
                    largest_size = size
                    largest_file = filepath
        
        if largest_file:
            processed_file = self.process_downloaded_dataset(largest_file)
            return processed_file
        else:
            return downloaded_files[0] if downloaded_files else None

if __name__ == "__main__":
    downloader = KaggleDatasetDownloader()
    result = downloader.run()
    
    if result:
        print(f"\n🎉 Dataset preparation completed!")
        print(f"📁 Large dataset ready at: {result}")
    else:
        print("\n❌ Dataset preparation failed")