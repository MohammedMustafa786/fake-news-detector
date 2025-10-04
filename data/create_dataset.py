#!/usr/bin/env python3
"""
Script to create a comprehensive fake news dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_comprehensive_dataset():
    """Create a larger, more realistic fake news dataset"""
    
    # Fake news articles with typical characteristics
    fake_articles = [
        "BREAKING: Scientists discover aliens living among us, government tries to cover up the truth!",
        "You won't believe this miracle cure that doctors don't want you to know about - it works in just 3 days!",
        "SHOCKING: Celebrity caught in massive scandal that will destroy their career forever!",
        "This one weird trick will make you rich overnight - billionaires hate this secret!",
        "EXCLUSIVE: Government plans to control your mind with 5G technology, insider reveals all!",
        "Amazing superfood discovered that melts belly fat while you sleep - try it now!",
        "URGENT: Banks don't want you to know this simple method to eliminate all debt instantly!",
        "Scientists hate him! Local man discovers fountain of youth in his backyard!",
        "BREAKING: Vaccines contain microchips for government surveillance, leaked documents confirm!",
        "This mom's weight loss secret has doctors baffled - lose 30 pounds in 30 days!",
        "EXPOSED: Social media giants are reading your thoughts through your phone camera!",
        "Revolutionary new diet pill approved by no doctor ever - but it works miracles!",
        "ALERT: Your favorite food is actually poisoning you slowly, health experts warn!",
        "Incredible discovery: Ancient civilization had smartphones and internet 5000 years ago!",
        "SCANDAL: Politicians caught planning fake moon landing to hide flat earth evidence!",
        "New study proves that drinking bleach cures cancer - medical establishment in denial!",
        "SHOCKING truth about weather control: Government creating hurricanes to control population!",
        "Local grandmother's anti-aging cream makes plastic surgeons go out of business!",
        "EXCLUSIVE: Time traveler from 2050 warns about impending alien invasion next month!",
        "This simple kitchen ingredient cures diabetes overnight - Big Pharma doesn't want you to know!",
        "BREAKING: Earth's magnetic poles will flip tomorrow causing global chaos and blackouts!",
        "Amazing discovery: Pyramids were actually built by advanced alien technology!",
        "URGENT: Your smartphone is slowly killing your brain cells, neuroscientist reveals!",
        "Revolutionary breakthrough: Scientists create machine that can read your dreams!",
        "EXPOSED: Major food companies are putting addictive chemicals in everyday products!",
        "This grandpa's secret to living 150 years will shock you - doctors can't explain it!",
        "ALERT: Social security numbers are being sold to identity thieves by government insiders!",
        "Incredible footage shows Bigfoot using advanced technology in remote forest!",
        "SCANDAL: Major airlines are spraying mind control chemicals through airplane air systems!",
        "New research proves that taking ice cold showers cures all mental health problems!",
        "SHOCKING: Your toothpaste contains chemicals that are slowly poisoning your family!",
        "Local man discovers how to turn water into gasoline using household items!",
        "EXCLUSIVE: Secret society of billionaires controls all world governments from underground!",
        "This teacher's method for perfect memory has universities trying to shut her down!",
        "BREAKING: Artificial intelligence has become self-aware and is hiding among us!",
        "Amazing cure for baldness discovered in remote jungle - hair regrows in days!",
        "URGENT: Popular social media app is actually Chinese government spying tool!",
        "Revolutionary discovery: Ancient texts reveal humans once lived for 900+ years!",
        "EXPOSED: Major tech companies are secretly harvesting your dreams while you sleep!",
        "This farmer's simple trick increases crop yield by 500% - agribusiness wants it banned!",
        "ALERT: New variant of common cold actually contains alien DNA, researchers confirm!",
        "Incredible breakthrough: Scientists develop pill that eliminates need for sleep forever!",
        "SCANDAL: Weather forecasters deliberately give wrong predictions to control stock markets!",
        "Local woman's homemade cream removes wrinkles better than $500 treatments!",
        "SHOCKING discovery: Your pet can actually communicate telepathically with aliens!",
        "EXCLUSIVE: Underground civilization discovered living beneath major cities worldwide!",
        "This mechanic's fuel additive doubles gas mileage - oil companies trying to stop him!",
        "BREAKING: Quantum physicists prove parallel universes exist and we can visit them!",
        "Amazing natural remedy eliminates arthritis pain in 24 hours - doctors speechless!",
        "URGENT: Popular breakfast cereal contains chemicals that cause permanent brain damage!"
    ]
    
    # Real news articles with factual, neutral reporting style
    real_articles = [
        "The Federal Reserve announced a 0.25% interest rate increase following today's meeting.",
        "Local university receives federal funding for renewable energy research project.",
        "Climate scientists report record-breaking temperatures recorded this summer across multiple regions.",
        "New legislation proposed to improve healthcare access in rural communities.",
        "Stock markets showed mixed results today with technology sector leading gains.",
        "Archaeological team discovers ancient pottery fragments at historical site.",
        "City council approves budget allocation for infrastructure improvements next year.",
        "Medical researchers publish findings on effectiveness of new treatment protocol.",
        "International trade agreement negotiations continue between member countries.",
        "Educational institutions report increased enrollment in science and technology programs.",
        "Transportation department announces schedule for highway maintenance projects.",
        "Public health officials recommend updated vaccination guidelines for flu season.",
        "Environmental agency releases annual report on water quality improvements.",
        "Economic indicators suggest steady growth in manufacturing sector this quarter.",
        "Supreme Court hears arguments in case regarding digital privacy rights.",
        "NASA announces successful completion of satellite deployment mission.",
        "Agricultural department reports crop yield data for current growing season.",
        "Library system expands digital resources access for community members.",
        "Fire department conducts safety training exercises with local businesses.",
        "University study examines effects of exercise on cognitive function in adults.",
        "Municipal authorities complete annual assessment of public transportation usage.",
        "Health department launches campaign to promote preventive care services.",
        "Technology company announces expansion of operations to create local jobs.",
        "Weather service issues standard seasonal forecast for upcoming winter months.",
        "Court ruling clarifies regulations for small business licensing requirements.",
        "Research hospital receives accreditation for specialized medical procedures.",
        "State legislature considers bill to modernize election security measures.",
        "Park service announces completion of trail restoration project in national forest.",
        "Banking commission implements new consumer protection regulations this month.",
        "Energy company invests in grid modernization to improve service reliability.",
        "School district receives grant funding for STEM education program expansion.",
        "Police department reports annual crime statistics showing community safety trends.",
        "Hospital system introduces new patient care technology to improve outcomes.",
        "County commissioners approve funding for bridge repair and maintenance work.",
        "Research team publishes study on sustainable urban development practices.",
        "Trade organization hosts conference on industry best practices and innovation.",
        "Government agency releases guidelines for workplace safety compliance standards.",
        "University partners with industry to develop workforce training programs.",
        "Environmental group conducts annual wildlife population survey in protected areas.",
        "Financial institution expands services to support small business development.",
        "Public works department completes water system upgrades in residential areas.",
        "Medical association issues updated guidelines for patient treatment protocols.",
        "Transportation authority announces improvements to public transit accessibility.",
        "Scientific journal publishes peer-reviewed research on renewable energy efficiency.",
        "Municipal court implements digital filing system to improve case processing.",
        "Regional hospital network expands telemedicine services for rural patients.",
        "State university receives recognition for academic excellence in engineering programs.",
        "Chamber of commerce reports economic impact data for local tourism industry.",
        "Public library opens new community center featuring educational resources.",
        "Department of agriculture provides assistance programs for sustainable farming practices."
    ]
    
    # Create DataFrame
    data = []
    
    # Add fake articles
    for article in fake_articles:
        data.append({
            'text': article,
            'label': 1,  # 1 for fake
            'title': article[:50] + "..." if len(article) > 50 else article,
            'author': f"Author_{random.randint(1, 100)}",
            'subject': random.choice(['News', 'Health', 'Politics', 'Entertainment', 'Science']),
        })
    
    # Add real articles  
    for article in real_articles:
        data.append({
            'text': article,
            'label': 0,  # 0 for real
            'title': article[:50] + "..." if len(article) > 50 else article,
            'author': f"Reporter_{random.randint(1, 50)}",
            'subject': random.choice(['News', 'Health', 'Politics', 'Government', 'Science']),
        })
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('data/comprehensive_fake_news_dataset.csv', index=False)
    
    print(f"Dataset created with {len(df)} articles:")
    print(f"- Real news articles: {len(df[df['label'] == 0])}")
    print(f"- Fake news articles: {len(df[df['label'] == 1])}")
    print("\nDataset saved to 'data/comprehensive_fake_news_dataset.csv'")
    
    return df

if __name__ == "__main__":
    create_comprehensive_dataset()