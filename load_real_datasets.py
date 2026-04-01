#!/usr/bin/env python3
"""
Download and process real fact-checking datasets to replace synthetic data.

This addresses the data leakage issue by using authentic misinformation vs.
factual claims from real fact-checking organizations.

REAL DATASETS AVAILABLE:
1. LIAR - 12.8K political statements (2016-2017)
2. FEVER - 180K claims with evidence
3. Climate Feedback - 2K+ climate articles with expert fact-checks
4. Multi-FC - Multi-lingual fact-checking (~35K claims)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import json
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealDatasetLoader:
    """Load and preprocess real fact-checking datasets."""
    
    def __init__(self, data_dir: Path = Path('data/raw')):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_liar_dataset(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load LIAR dataset of political statements.
        
        Download from: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
        
        Format:
        - id, label, statement, subject, speaker, job-title, state, party,
          barely-true-count, false-count, half-true-count, mostly-true-count,
          pants-fire-count, context
        
        Labels: pants-fire, false, mostly-false, half-true, mostly-true, true
        
        Map to binary:
        - FALSE: pants-fire, false, mostly-false (0)
        - TRUE: mostly-true, true, half-true (1)
        """
        
        if filepath is None:
            filepath = self.data_dir / 'liar_dataset' / 'train.tsv'
        
        if not filepath.exists():
            print(f"\n❌ LIAR dataset not found at {filepath}")
            print("Download from: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
            print("Extract to: data/raw/liar_dataset/")
            return None
        
        logger.info(f"Loading LIAR dataset from {filepath}")
        
        try:
            # Load TSV
            df = pd.read_csv(filepath, sep='\t', header=None)
            
            # Assign column names based on LIAR format
            df.columns = [
                'id', 'label', 'statement', 'subject', 'speaker',
                'job_title', 'state', 'party', 'barely_true',
                'false_count', 'half_true', 'mostly_true',
                'pants_fire', 'context'
            ]
            
            # Map labels to binary (0=FALSE, 1=TRUE)
            label_map = {
                'pants-fire': 0, 'false': 0, 'mostly-false': 0,
                'half-true': 1, 'mostly-true': 1, 'true': 1
            }
            df['binary_label'] = df['label'].map(label_map)
            
            # Select relevant columns
            df = df[[
                'statement', 'binary_label', 'subject',
                'speaker', 'context'
            ]].copy()
            
            df.columns = ['claim', 'label', 'topic', 'source_speaker', 'context']
            df['source'] = 'LIAR'
            
            logger.info(f"✓ Loaded {len(df):,} claims from LIAR")
            logger.info(f"  - FALSE (0): {(df['label']==0).sum():,}")
            logger.info(f"  - TRUE (1): {(df['label']==1).sum():,}")
            logger.info(f"  - Topics: {df['topic'].nunique()} unique")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading LIAR: {e}")
            return None
    
    def load_fever_dataset(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load FEVER (Fact Extraction and Verification) dataset.
        
        Download from: http://fever.ai/
        
        Format: JSON lines with claims and supporting/refuting evidence
        
        Map to binary:
        - FALSE: REFUTES (0)
        - TRUE: SUPPORTS (1)
        - Drop: NOT ENOUGH INFO
        """
        
        if filepath is None:
            filepath = self.data_dir / 'fever' / 'train.jsonl'
        
        if not filepath.exists():
            print(f"\n❌ FEVER dataset not found at {filepath}")
            print("Download from: http://fever.ai/")
            return None
        
        logger.info(f"Loading FEVER dataset from {filepath}")
        
        try:
            claims = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    record = json.loads(line.strip())
                    
                    # Only include SUPPORTS/REFUTES (drop NOT ENOUGH INFO)
                    if record.get('label') not in ['SUPPORTS', 'REFUTES']:
                        continue
                    
                    label = 1 if record['label'] == 'SUPPORTS' else 0
                    evidence_text = " | ".join([
                        ev[2] if len(ev) > 2 else ""
                        for ev_list in record.get('evidence', [])
                        for ev in ev_list if isinstance(ev, (list, tuple))
                    ])[:200]  # Limit length
                    
                    claims.append({
                        'claim': record['claim'],
                        'label': label,
                        'topic': 'general',
                        'evidence': evidence_text,
                        'source': 'FEVER'
                    })
            
            df = pd.DataFrame(claims)
            logger.info(f"✓ Loaded {len(df):,} claims from FEVER")
            logger.info(f"  - FALSE (0): {(df['label']==0).sum():,}")
            logger.info(f"  - TRUE (1): {(df['label']==1).sum():,}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading FEVER: {e}")
            return None
    
    def combine_datasets(self, datasets: dict) -> pd.DataFrame:
        """Combine multiple datasets into single training set."""
        
        logger.info("\nCombining datasets...")
        
        all_data = []
        for name, df in datasets.items():
            if df is not None:
                logger.info(f"  - {name}: {len(df)} claims")
                all_data.append(df)
        
        if not all_data:
            logger.error("No datasets available")
            return None
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Shuffle
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"\n✓ Combined dataset: {len(combined):,} claims")
        logger.info(f"  - FALSE (0): {(combined['label']==0).sum():,}")
        logger.info(f"  - TRUE (1): {(combined['label']==1).sum():,}")
        logger.info(f"  - Class balance: {100*(combined['label']==0).sum()/len(combined):.1f}% FALSE")
        
        return combined
    
    def create_test_set_from_kaggle(self) -> Optional[pd.DataFrame]:
        """
        Create a balanced test set from alternative source for model validation.
        
        Alternative sources to avoid data leakage:
        - FakeNewsNet dataset (Kaggle)
        - MultiWOZ dataset
        - Stance detection datasets
        """
        
        print("""
Alternative Test Datasets:
================================

1. FakeNewsNet (Kaggle)
   - URL: https://www.kaggle.com/datasets/jruvika/fake-news-detection
   - ~400K articles from PolitiFact and GossipCop
   
2. Rumor Verification (Twitter)
   - URL: https://alt.qcri.org/semeval2017/task8/
   - ~5K rumors with verification labels
   
3. Satire Detection (NELA-CORE)
   - URL: https://github.com/mitresomneuk/nela-core-2018
   - ~1K satire vs. mainstream articles
        """)
        
        return None


def main():
    """Main execution - load real datasets."""
    
    print("\n" + "="*70)
    print("REAL FACT-CHECKING DATASETS")
    print("="*70)
    print("""
REPLACING SYNTHETIC DATA TO FIX 100% ACCURACY ISSUE:

The previous synthetic dataset was created by resampling 6-12 items,
causing data leakage and memorization. Real datasets provide:

✓ Authentic linguistic diversity
✓ Real-world confounding factors
✓ Complex claim variations
✓ Natural class imbalance challenges
✓ Domain-specific vocabulary patterns
    """)
    
    loader = RealDatasetLoader()
    
    datasets = {}
    
    # Try to load LIAR
    liar_df = loader.load_liar_dataset()
    if liar_df is not None:
        datasets['LIAR'] = liar_df
    
    # Try to load FEVER
    fever_df = loader.load_fever_dataset()
    if fever_df is not None:
        datasets['FEVER'] = fever_df
    
    # Combine if available
    if datasets:
        combined = loader.combine_datasets(datasets)
        
        # Save
        output_path = Path('data/processed') / 'real_factchecking_dataset.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved combined dataset to {output_path}")
    else:
        print("\n" + "="*70)
        print("DOWNLOAD INSTRUCTIONS")
        print("="*70)
        print("""
Option 1: Download LIAR Dataset
  1. Visit: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
  2. Extract to: data/raw/liar_dataset/
  3. Run this script again

Option 2: Download FEVER Dataset
  1. Visit: http://fever.ai/
  2. Download training/dev splits (JSON Lines format)
  3. Extract to: data/raw/fever/
  4. Run this script again

Option 3: Use Kaggle Datasets
  1. Install: pip install kaggle
  2. Setup API key: https://www.kaggle.com/account
  3. Download: kaggle datasets download -d jruvika/fake-news-detection
        """)
    
    # Test set info
    loader.create_test_set_from_kaggle()


if __name__ == '__main__':
    main()
