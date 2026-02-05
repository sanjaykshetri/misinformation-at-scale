#!/usr/bin/env python3
"""
Generate realistic synthetic Reddit comments for testing the ML pipeline.
Uses actual linguistic patterns to create realistic training data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path


# Realistic comment examples
MISINFORMATION_PATTERNS = [
    "did you see what they're really hiding about {}?",
    "the mainstream media won't tell you about {}",
    "wake up sheeple, {} is a hoax",
    "they don't want you to know about {}",
    "the government is covering up {}",
    "have you noticed how no one talks about {}?",
    "this is being censored everywhere because of {}",
    "don't believe what they tell you about {}",
    "the truth about {} will shock you",
    "they're lying to us about {}",
    "follow the money and you'll understand {}",
    "nobody questions {} anymore",
    "it's obvious if you just do your research on {}",
    "the elites don't want us knowing about {}",
    "big pharma is hiding {} from us",
    "social media is suppressing posts about {}",
    "the deep state is involved in {}",
    "this is exactly what they predicted with {}",
    "the simulation/matrix connection to {}",
    "historical documents prove {}",
]

MISINFORMATION_TOPICS = [
    "vaccines",
    "election fraud",
    "5G towers",
    "climate change",
    "moon landing",
    "flat earth",
    "JFK assassination",
    "9/11 truth",
    "area 51",
    "lizard people",
    "new world order",
    "chemtrails",
    "illuminati",
    "bill gates",
    "big pharma",
    "chemotherapy",
    "fiat currency",
    "ancient aliens",
]

CONTROL_COMMENTS = [
    "According to recent studies, {}",
    "The scientific evidence shows that {}",
    "Research indicates that {}",
    "Based on peer-reviewed data, {}",
    "Multiple studies confirm that {}",
    "The consensus among experts is that {}",
    "Data from the CDC shows that {}",
    "According to NASA, {}",
    "The WHO recommends {}",
    "Published research demonstrates that {}",
    "In a meta-analysis, scientists found that {}",
    "The evidence-based approach shows that {}",
    "Experts agree that {}",
    "Long-term studies show that {}",
    "Clinical trials indicate that {}",
    "The peer review process validates {}",
    "Longitudinal data suggests that {}",
    "Large-scale analysis reveals that {}",
    "Reproducible experiments show that {}",
    "The scientific method proves that {}",
]

CONTROL_TOPICS = [
    "climate change is real",
    "vaccines are safe",
    "the earth is round",
    "the moon landing was real",
    "COVID-19 is a virus",
    "5G is safe",
    "evolution is supported by evidence",
    "gravity works as described",
    "antibiotics kill bacteria",
    "the sun is 93 million miles away",
    "water boils at 100°C",
    "atoms exist",
    "photosynthesis is how plants make food",
    "electricity is safe when properly grounded",
    "germ theory is correct",
    "DNA carries genetic information",
    "the earth orbits the sun",
    "radiation follows inverse square law",
    "vaccines contain no tracking devices",
    "antidepressants help depression",
]


def generate_synthetic_comments(n_comments=10000, seed=42):
    """
    Generate realistic synthetic Reddit comments.
    
    Args:
        n_comments: Number of comments to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic comments
    """
    np.random.seed(seed)
    random.seed(seed)
    
    comments = []
    
    # Generate misinformation comments
    n_misinformation = n_comments // 2
    for _ in range(n_misinformation):
        pattern = random.choice(MISINFORMATION_PATTERNS)
        topic = random.choice(MISINFORMATION_TOPICS)
        body = pattern.format(topic)
        
        # Add noise/variations
        if random.random() > 0.5:
            body = body.upper()
        if random.random() > 0.8:
            body += " !!! Wake up!!!"
        
        comment = {
            'author': f'user_{random.randint(1000, 99999)}',
            'body': body,
            'created_utc': int((datetime.now() - timedelta(days=random.randint(0, 365))).timestamp()),
            'score': random.randint(-50, 100),
            'subreddit': random.choice(['conspiracy', 'theDonald', 'NoNewNormal']),
            'id': f'c_{random.randint(100000, 999999):06d}',
        }
        comments.append(comment)
    
    # Generate control (science) comments
    n_control = n_comments - n_misinformation
    for _ in range(n_control):
        pattern = random.choice(CONTROL_COMMENTS)
        topic = random.choice(CONTROL_TOPICS)
        body = pattern.format(topic)
        
        # Add variations
        if random.random() > 0.9:
            body += " [1] [2] [3]"  # Reference notation
        
        comment = {
            'author': f'user_{random.randint(1000, 99999)}',
            'body': body,
            'created_utc': int((datetime.now() - timedelta(days=random.randint(0, 365))).timestamp()),
            'score': random.randint(-10, 500),
            'subreddit': random.choice(['science', 'askscience', 'news']),
            'id': f'c_{random.randint(100000, 999999):06d}',
        }
        comments.append(comment)
    
    return pd.DataFrame(comments)


def main(output_file='data/raw/reddit_comments_2020.csv'):
    """Generate and save synthetic Reddit data."""
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Synthetic Reddit Comments")
    print("=" * 60)
    print("\nThis creates realistic training data based on:")
    print("  • Misinformation linguistic patterns")
    print("  • Science/control comment patterns")
    print("  • Realistic Reddit metadata")
    print()
    
    # Generate data
    print("Generating 10,000 comments...")
    df = generate_synthetic_comments(n_comments=10000)
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(df):,} comments to {output_file}\n")
    
    # Print summary
    print("Summary:")
    print(f"  Total comments: {len(df):,}")
    print(f"  Subreddits: {df['subreddit'].nunique()}")
    print(f"  Date range: {datetime.fromtimestamp(df['created_utc'].min()).date()} to {datetime.fromtimestamp(df['created_utc'].max()).date()}")
    print(f"  Avg score: {df['score'].mean():.1f}")
    print(f"  File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    print(f"\nComment distribution:")
    print(df['subreddit'].value_counts().to_string())
    print(f"\n✅ Data ready! Run: jupyter notebook notebooks/01_data_wrangling.ipynb\n")


if __name__ == '__main__':
    main()
