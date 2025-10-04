#!/usr/bin/env python3
"""
Multilingual Dataset Creator for Fake News Detection
Creates balanced datasets in Spanish, French, and Hindi with fact-checking corpora
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path
import json

class MultilingualDatasetCreator:
    """
    Creates comprehensive multilingual datasets for fake news detection.
    """
    
    def __init__(self):
        self.data_dir = Path("data/multilingual")
        self.data_dir.mkdir(exist_ok=True)
        
        # Language-specific fake news patterns
        self.fake_patterns = {
            'spanish': {
                'sensational_words': [
                    'EXCLUSIVO', 'IMPACTANTE', 'URGENTE', 'INCREÍBLE', 'ESCÁNDALO', 
                    'REVELADO', 'SECRETO', 'FILTRADO', 'BOMBAZO', 'ALERTA'
                ],
                'clickbait_phrases': [
                    'No podrás creer lo que pasó después',
                    'Los doctores odian este truco simple',
                    'Esto cambiará todo lo que sabías sobre',
                    'Los científicos no quieren que sepas esto',
                    'La verdad que te han estado ocultando',
                    'Lo que pasó después te sorprenderá'
                ],
                'fake_health_claims': [
                    'Esta especia común cura la diabetes de la noche a la mañana',
                    'Bebe esto antes de dormir para perder 15 kilos en una semana',
                    'Este ingrediente de cocina elimina todas las toxinas del cuerpo',
                    'Los médicos descubrieron este alimento que rejuvenece 20 años',
                    'Este ejercicio simple elimina el dolor de artritis para siempre'
                ]
            },
            'french': {
                'sensational_words': [
                    'EXCLUSIF', 'CHOQUANT', 'URGENT', 'INCROYABLE', 'SCANDALE',
                    'RÉVÉLÉ', 'SECRET', 'FUITE', 'BOMBE', 'ALERTE'
                ],
                'clickbait_phrases': [
                    'Vous ne croirez pas ce qui s\'est passé ensuite',
                    'Les médecins détestent cette astuce simple',
                    'Cela va changer tout ce que vous saviez sur',
                    'Les scientifiques ne veulent pas que vous sachiez cela',
                    'La vérité qu\'ils vous ont cachée',
                    'Ce qui s\'est passé ensuite va vous surprendre'
                ],
                'fake_health_claims': [
                    'Cette épice commune guérit le diabète du jour au lendemain',
                    'Buvez ceci avant de dormir pour perdre 15 kilos en une semaine',
                    'Cet ingrédient de cuisine élimine toutes les toxines du corps',
                    'Les médecins ont découvert cet aliment qui rajeunit de 20 ans',
                    'Cet exercice simple élimine la douleur arthritique pour toujours'
                ]
            },
            'hindi': {
                'sensational_words': [
                    'एक्सक्लूसिव', 'चौंकाने वाला', 'तुरंत', 'अविश्वसनीय', 'स्कैंडल',
                    'खुलासा', 'गुप्त', 'लीक', 'बम', 'अलर्ट'
                ],
                'clickbait_phrases': [
                    'आप विश्वास नहीं कर सकते कि आगे क्या हुआ',
                    'डॉक्टर इस सरल ट्रिक से नफरत करते हैं',
                    'यह आपको पता सब कुछ बदल देगा',
                    'वैज्ञानिक नहीं चाहते कि आप यह जानें',
                    'सच्चाई जो आपसे छुपाई गई है',
                    'जो आगे हुआ वह आपको आश्चर्यचकित कर देगा'
                ],
                'fake_health_claims': [
                    'यह आम मसाला रात भर में मधुमेह ठीक करता है',
                    'सोने से पहले यह पीकर एक सप्ताह में 15 किलो वजन कम करें',
                    'यह रसोई घर का तत्व शरीर के सभी विषाक्त पदार्थों को हटा देता है',
                    'डॉक्टरों ने इस भोजन की खोज की है जो 20 साल जवान बनाता है',
                    'यह सरल व्यायाम हमेशा के लिए गठिया दर्द को खत्म करता है'
                ]
            }
        }
        
        # Real news templates in different languages
        self.real_templates = {
            'spanish': [
                'El Ministerio de {} anunció nuevas políticas de {} tras una extensa investigación.',
                'Según el último informe de {}, los indicadores económicos muestran un crecimiento del {}% en el sector {}.',
                'Los investigadores de la Universidad de {} publicaron hallazgos sobre {} en la revista {}.',
                'El Departamento de {} emitió nuevas directrices para el cumplimiento de seguridad en {}.',
                'Las autoridades locales recibieron financiamiento federal para programas de desarrollo en {}.',
            ],
            'french': [
                'Le Ministère de {} a annoncé de nouvelles politiques de {} après des recherches approfondies.',
                'Selon le dernier rapport de {}, les indicateurs économiques montrent une croissance de {}% dans le secteur {}.',
                'Les chercheurs de l\'Université de {} ont publié des résultats sur {} dans la revue {}.',
                'Le Département de {} a émis de nouvelles directives pour la conformité de sécurité en {}.',
                'Les autorités locales ont reçu un financement fédéral pour des programmes de développement en {}.',
            ],
            'hindi': [
                '{} मंत्रालय ने व्यापक अनुसंधान के बाद {} की नई नीतियों की घोषणा की।',
                '{} की नवीनतम रिपोर्ट के अनुसार, आर्थिक संकेतक {} क्षेत्र में {}% की वृद्धि दिखाते हैं।',
                '{} विश्वविद्यालय के शोधकर्ताओं ने {} पत्रिका में {} पर निष्कर्ष प्रकाशित किए।',
                '{} विभाग ने {} में सुरक्षा अनुपालन के लिए नए दिशानिर्देश जारी किए।',
                'स्थानीय अधिकारियों को {} में विकास कार्यक्रमों के लिए संघीय वित्त पोषण प्राप्त हुआ।',
            ]
        }
        
        # Language-specific vocabulary
        self.vocabulary = {
            'spanish': {
                'departments': ['Salud', 'Educación', 'Transporte', 'Agricultura', 'Comercio', 'Energía'],
                'policies': ['salud', 'educación', 'medio ambiente', 'economía', 'infraestructura', 'seguridad'],
                'universities': ['Barcelona', 'Madrid', 'Valencia', 'Sevilla', 'Salamanca'],
                'subjects': ['medicina', 'ingeniería', 'economía', 'psicología', 'biología', 'física'],
                'sectors': ['tecnología', 'salud', 'manufactura', 'agricultura', 'energía', 'finanzas'],
                'percentages': ['2.3', '1.8', '3.7', '0.9', '4.2', '1.5'],
                'journals': ['Natura', 'Ciencia', 'Investigación Médica']
            },
            'french': {
                'departments': ['Santé', 'Éducation', 'Transport', 'Agriculture', 'Commerce', 'Énergie'],
                'policies': ['santé', 'éducation', 'environnement', 'économie', 'infrastructure', 'sécurité'],
                'universities': ['Sorbonne', 'Lyon', 'Marseille', 'Bordeaux', 'Strasbourg'],
                'subjects': ['médecine', 'ingénierie', 'économie', 'psychologie', 'biologie', 'physique'],
                'sectors': ['technologie', 'santé', 'fabrication', 'agriculture', 'énergie', 'finances'],
                'percentages': ['2.3', '1.8', '3.7', '0.9', '4.2', '1.5'],
                'journals': ['Nature', 'Science', 'Recherche Médicale']
            },
            'hindi': {
                'departments': ['स्वास्थ्य', 'शिक्षा', 'परिवहन', 'कृषि', 'वाणिज्य', 'ऊर्जा'],
                'policies': ['स्वास्थ्य', 'शिक्षा', 'पर्यावरण', 'अर्थव्यवस्था', 'अवसंरचना', 'सुरक्षा'],
                'universities': ['दिल्ली', 'मुंबई', 'कोलकाता', 'चेन्नई', 'बेंगलुरु'],
                'subjects': ['चिकित्सा', 'इंजीनियरिंग', 'अर्थशास्त्र', 'मनोविज्ञान', 'जीवविज्ञान', 'भौतिकी'],
                'sectors': ['प्रौद्योगिकी', 'स्वास्थ्य', 'विनिर्माण', 'कृषि', 'ऊर्जा', 'वित्त'],
                'percentages': ['2.3', '1.8', '3.7', '0.9', '4.2', '1.5'],
                'journals': ['प्रकृति', 'विज्ञान', 'चिकित्सा अनुसंधान']
            }
        }
    
    def generate_fake_news(self, language, count=1000):
        """Generate fake news articles in specified language."""
        patterns = self.fake_patterns[language]
        fake_articles = []
        
        # Generate different types of fake news
        for _ in range(count // 3):
            # Sensational news
            sensational = random.choice(patterns['sensational_words'])
            clickbait = random.choice(patterns['clickbait_phrases'])
            
            if language == 'spanish':
                article = f"{sensational}: Nueva evidencia revela la verdad oculta. {clickbait} esta información explosiva. Los medios principales se niegan a informar sobre esto."
            elif language == 'french':
                article = f"{sensational}: De nouvelles preuves révèlent la vérité cachée. {clickbait} ces informations explosives. Les médias principaux refusent de rendre compte de cela."
            else:  # hindi
                article = f"{sensational}: नए सबूत छुपे हुए सच का खुलासा करते हैं। {clickbait} यह विस्फोटक जानकारी। मुख्यधारा की मीडिया इस पर रिपोर्ट करने से इनकार करती है।"
            
            fake_articles.append(article)
        
        # Health misinformation
        for _ in range(count // 3):
            claim = random.choice(patterns['fake_health_claims'])
            sensational = random.choice(patterns['sensational_words'])
            clickbait = random.choice(patterns['clickbait_phrases'])
            
            if language == 'spanish':
                article = f"{sensational}: {claim} Las grandes farmacéuticas no quieren que descubras este remedio natural. {clickbait} esta cura milagrosa."
            elif language == 'french':
                article = f"{sensational}: {claim} Les grandes entreprises pharmaceutiques ne veulent pas que vous découvriez ce remède naturel. {clickbait} ce remède miracle."
            else:  # hindi
                article = f"{sensational}: {claim} बड़ी दवा कंपनियां नहीं चाहतीं कि आप इस प्राकृतिक उपचार की खोज करें। {clickbait} इस चमत्कारी इलाज।"
            
            fake_articles.append(article)
        
        # Conspiracy theories
        for _ in range(count // 3):
            sensational = random.choice(patterns['sensational_words'])
            clickbait = random.choice(patterns['clickbait_phrases'])
            
            if language == 'spanish':
                article = f"{sensational}: Información privilegiada revela que las empresas tecnológicas están controlando secretamente nuestros pensamientos. {clickbait} esta tecnología. Esta información filtrada muestra cuán profundo es realmente el engaño."
            elif language == 'french':
                article = f"{sensational}: Un initié révèle que les entreprises technologiques contrôlent secrètement nos pensées. {clickbait} cette technologie. Ces informations divulguées montrent à quel point la tromperie est vraiment profonde."
            else:  # hindi
                article = f"{sensational}: अंदरूनी सूत्र से पता चलता है कि तकनीकी कंपनियां गुप्त रूप से हमारे विचारों को नियंत्रित कर रही हैं। {clickbait} यह तकनीक। यह लीक हुई जानकारी दिखाती है कि धोखा वास्तव में कितना गहरा है।"
            
            fake_articles.append(article)
        
        return fake_articles
    
    def generate_real_news(self, language, count=1000):
        """Generate real news articles in specified language."""
        templates = self.real_templates[language]
        vocab = self.vocabulary[language]
        real_articles = []
        
        for _ in range(count):
            template = random.choice(templates)
            
            try:
                if language == 'spanish' or language == 'french':
                    article = template.format(
                        random.choice(vocab['departments']),
                        random.choice(vocab['policies']),
                        random.choice(vocab['universities']),
                        random.choice(vocab['percentages']),
                        random.choice(vocab['sectors']),
                        random.choice(vocab['subjects']),
                        random.choice(vocab['journals'])
                    )
                else:  # hindi
                    article = template.format(
                        random.choice(vocab['departments']),
                        random.choice(vocab['policies']),
                        random.choice(vocab['universities']),
                        random.choice(vocab['subjects']),
                        random.choice(vocab['journals']),
                        random.choice(vocab['percentages']),
                        random.choice(vocab['sectors'])
                    )
            except:
                # Fallback if template formatting fails
                article = template
            
            real_articles.append(article)
        
        return real_articles
    
    def create_language_dataset(self, language, fake_count=2000, real_count=2000):
        """Create a complete dataset for one language."""
        print(f"🌍 Creating {language.title()} dataset...")
        
        # Generate articles
        fake_articles = self.generate_fake_news(language, fake_count)
        real_articles = self.generate_real_news(language, real_count)
        
        # Create dataset
        data = []
        
        # Add fake articles
        for i, article in enumerate(fake_articles):
            data.append({
                'text': article,
                'label': 1,  # fake
                'title': article[:100] + "..." if len(article) > 100 else article,
                'author': f'FakeAuthor_{language}_{i % 100}',
                'subject': random.choice(['Politics', 'Health', 'Technology', 'Science', 'Economy']),
                'language': language
            })
        
        # Add real articles
        for i, article in enumerate(real_articles):
            data.append({
                'text': article,
                'label': 0,  # real
                'title': article[:100] + "..." if len(article) > 100 else article,
                'author': f'Reporter_{language}_{i % 50}',
                'subject': random.choice(['Politics', 'Health', 'Technology', 'Science', 'Economy']),
                'language': language
            })
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save dataset
        filename = self.data_dir / f'{language}_fake_news_dataset.csv'
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"   ✅ {language.title()} dataset saved: {filename}")
        print(f"   📊 Total articles: {len(df):,}")
        print(f"   📰 Real articles: {len(df[df['label'] == 0]):,}")
        print(f"   📰 Fake articles: {len(df[df['label'] == 1]):,}")
        print(f"   💾 File size: {filename.stat().st_size / 1024 / 1024:.1f} MB")
        
        return df, filename
    
    def create_combined_multilingual_dataset(self):
        """Create a combined dataset with all languages."""
        print(f"\n🌐 Creating combined multilingual dataset...")
        
        # Load individual language datasets
        combined_data = []
        
        for language in ['spanish', 'french', 'hindi']:
            filename = self.data_dir / f'{language}_fake_news_dataset.csv'
            if filename.exists():
                df = pd.read_csv(filename, encoding='utf-8')
                combined_data.append(df)
                print(f"   📁 Loaded {language}: {len(df):,} articles")
        
        # Combine all datasets
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Save combined dataset
            combined_filename = self.data_dir / 'multilingual_fake_news_dataset.csv'
            combined_df.to_csv(combined_filename, index=False, encoding='utf-8')
            
            print(f"\n✅ Combined multilingual dataset created!")
            print(f"   📁 Saved to: {combined_filename}")
            print(f"   📊 Total articles: {len(combined_df):,}")
            print(f"   🌍 Languages: {len(combined_df['language'].unique())}")
            print(f"   📰 Real articles: {len(combined_df[combined_df['label'] == 0]):,}")
            print(f"   📰 Fake articles: {len(combined_df[combined_df['label'] == 1]):,}")
            print(f"   💾 File size: {combined_filename.stat().st_size / 1024 / 1024:.1f} MB")
            
            # Language distribution
            print(f"\n🌐 Language Distribution:")
            for lang in combined_df['language'].unique():
                count = len(combined_df[combined_df['language'] == lang])
                percentage = count / len(combined_df) * 100
                print(f"   {lang.title()}: {count:,} articles ({percentage:.1f}%)")
            
            return combined_df, combined_filename
        else:
            print("❌ No language datasets found to combine")
            return None, None
    
    def create_all_datasets(self):
        """Create all multilingual datasets."""
        print("🚀" * 20)
        print("  MULTILINGUAL FAKE NEWS DATASET CREATOR")
        print("🚀" * 20)
        print("Creating datasets in Spanish, French, and Hindi...\n")
        
        # Create individual language datasets
        datasets = {}
        for language in ['spanish', 'french', 'hindi']:
            df, filename = self.create_language_dataset(language)
            datasets[language] = (df, filename)
        
        # Create combined dataset
        combined_df, combined_filename = self.create_combined_multilingual_dataset()
        
        # Summary
        print(f"\n🎉 Multilingual Dataset Creation Complete!")
        print(f"   📁 Individual datasets: 3 languages")
        print(f"   📁 Combined dataset: {len(combined_df):,} articles" if combined_df is not None else "❌ Combined dataset failed")
        print(f"   💾 Storage location: {self.data_dir}/")
        
        return datasets, combined_df

def main():
    """Main execution function."""
    creator = MultilingualDatasetCreator()
    datasets, combined_df = creator.create_all_datasets()
    
    if combined_df is not None:
        print(f"\n✅ Success! Multilingual datasets ready for training.")
        print(f"📈 Next steps:")
        print(f"   1. Train model with multilingual data")
        print(f"   2. Test Spanish confidence improvement")
        print(f"   3. Evaluate French and Hindi performance")
    else:
        print(f"\n❌ Dataset creation failed!")

if __name__ == "__main__":
    main()