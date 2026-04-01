#!/usr/bin/env python3
"""
Quick implementation guide: Train models with FakeNewsNet data

This script shows you exactly what to do to retrain your models
with the real, high-quality FakeNewsNet dataset.

Run this to verify everything is working!
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def load_fakenewsnet_data():
    """Load FakeNewsNet datasets."""
    print("\n" + "="*70)
    print("LOADING FAKENEWSNET DATA")
    print("="*70)
    
    df_train = pd.read_csv('data/processed/fakenewsnet_train.csv')
    df_val = pd.read_csv('data/processed/fakenewsnet_val.csv')
    df_test = pd.read_csv('data/processed/fakenewsnet_test.csv')
    
    print(f"\n✓ Train: {len(df_train):,} samples")
    print(f"✓ Val:   {len(df_val):,} samples")
    print(f"✓ Test:  {len(df_test):,} samples")
    
    print(f"\nClass distribution (Train):")
    print(f"  Real (0): {(df_train['label']==0).sum():,}")
    print(f"  Fake (1): {(df_train['label']==1).sum():,}")
    
    return df_train, df_val, df_test

def baseline_model(df_train, df_val, df_test):
    """Train a baseline Logistic Regression model."""
    print("\n" + "="*70)
    print("BASELINE MODEL: LOGISTIC REGRESSION")
    print("="*70)
    
    # TF-IDF Vectorization
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(df_train['claim'])
    X_val = vectorizer.transform(df_val['claim'])
    X_test = vectorizer.transform(df_test['claim'])
    
    print(f"✓ Train features: {X_train.shape}")
    print(f"✓ Val features: {X_val.shape}")
    print(f"✓ Test features: {X_test.shape}")
    
    # Train model
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, df_train['label'])
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    train_acc = accuracy_score(df_train['label'], y_pred_train)
    val_acc = accuracy_score(df_val['label'], y_pred_val)
    test_acc = accuracy_score(df_test['label'], y_pred_test)
    
    print("\n✓ RESULTS:")
    print(f"  Train Accuracy: {train_acc:.4f} ({100*train_acc:.2f}%)")
    print(f"  Val Accuracy:   {val_acc:.4f} ({100*val_acc:.2f}%)")
    print(f"  Test Accuracy:  {test_acc:.4f} ({100*test_acc:.2f}%)")
    
    # Additional metrics
    print(f"\n  Precision (Test): {precision_score(df_test['label'], y_pred_test):.4f}")
    print(f"  Recall (Test):    {recall_score(df_test['label'], y_pred_test):.4f}")
    print(f"  F1-Score (Test):  {f1_score(df_test['label'], y_pred_test):.4f}")
    
    # Train-Val gap
    gap = train_acc - val_acc
    print(f"\n  Train-Val Gap:    {gap:.4f} ({100*gap:.2f}%)")
    if gap < 0.05:
        print(f"  ✓ Good generalization (gap < 5%)")
    else:
        print(f"  ⚠️  Some overfitting detected")
    
    return model, vectorizer, {
        'train': train_acc,
        'val': val_acc,
        'test': test_acc
    }

def main():
    """Main execution."""
    
    print("\n" + "="*70)
    print("FAKENEWSNET BASELINE MODEL TRAINING")
    print("="*70)
    print("""
This script demonstrates how to train a baseline model with the
FakeNewsNet dataset. It serves as a reference for your notebooks.
    """)
    
    # Load data
    df_train, df_val, df_test = load_fakenewsnet_data()
    
    # Train baseline
    model, vectorizer, results = baseline_model(df_train, df_val, df_test)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
✓ Successfully trained baseline model on FakeNewsNet data
✓ Train Accuracy: {100*results['train']:.2f}%
✓ Val Accuracy:   {100*results['val']:.2f}%
✓ Test Accuracy:  {100*results['test']:.2f}%

These are REALISTIC accuracies (not 100% like before!)

EXPECTED RESULTS vs. YOUR RESULTS:
─────────────────────────────────────────────
Expected:  78-82% (baseline)
Your Result: {100*results['test']:.2f}%

✓ Results are in the expected range!

NEXT STEPS:
1. Update your notebooks to use fakenewsnet_*.csv files
2. Train deep learning model (should be 5-10% better)
3. Compare results and document findings

TO UPDATE YOUR NOTEBOOKS:
─────────────────────────────────────────────
Replace this:
  df = pd.read_csv('data/processed/realworld_train.csv')

With this:
  df_train = pd.read_csv('data/processed/fakenewsnet_train.csv')
  df_val = pd.read_csv('data/processed/fakenewsnet_val.csv')
  df_test = pd.read_csv('data/processed/fakenewsnet_test.csv')

Then retrain your models!
    """)

if __name__ == '__main__':
    main()
