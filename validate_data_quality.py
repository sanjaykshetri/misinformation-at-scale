#!/usr/bin/env python3
"""
Data Quality Validation Script

Detect and diagnose data leakage issues:
- Check uniqueness ratio
- Verify train/val/test separation
- Identify exact duplicates
- Validate class balance
- Provide quality metrics

Run this on any dataset before training models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, Optional


class DataQualityValidator:
    """Validate dataset quality and detect data leakage."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def _log(self, message: str, level: str = "INFO"):
        """Pretty print log messages."""
        if not self.verbose:
            return
        
        symbols = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✓ ",
            "WARNING": "⚠️  ",
            "ERROR": "❌ ",
            "HEADER": "━━",
        }
        
        print(f"{symbols.get(level, '  ')} {message}")
    
    def validate_dataset(self, filepath: str, text_col: str = 'claim', label_col: str = 'label') -> dict:
        """
        Comprehensive dataset validation.
        
        Returns:
            dict: Quality metrics and warnings
        """
        
        self._log("="*70, "HEADER")
        self._log("DATA QUALITY VALIDATION", "HEADER")
        self._log("="*70, "HEADER")
        
        # Load data
        try:
            df = pd.read_csv(filepath)
            self._log(f"Loaded {filepath}")
        except Exception as e:
            self._log(f"Failed to load {filepath}: {e}", "ERROR")
            return {}
        
        results = {
            'filepath': filepath,
            'passed_checks': [],
            'warnings': [],
            'errors': []
        }
        
        # 1. Basic structure
        self._log("\n1. DATASET STRUCTURE", "HEADER")
        self._log(f"Total rows: {len(df):,}")
        self._log(f"Total columns: {len(df.columns)}")
        
        if text_col not in df.columns:
            results['errors'].append(f"Missing required column: {text_col}")
        if label_col not in df.columns:
            results['errors'].append(f"Missing required column: {label_col}")
        
        results['total_rows'] = len(df)
        results['total_columns'] = len(df.columns)
        
        # 2. Uniqueness Analysis (Critical for data leakage)
        self._log("\n2. UNIQUENESS ANALYSIS", "HEADER")
        
        unique_count = df[text_col].nunique()
        uniqueness_ratio = unique_count / len(df)
        
        self._log(f"Unique {text_col}: {unique_count:,} out of {len(df):,}")
        self._log(f"Uniqueness ratio: {100*uniqueness_ratio:.2f}%")
        
        results['unique_count'] = unique_count
        results['uniqueness_ratio'] = uniqueness_ratio
        
        # Thresholds
        if uniqueness_ratio < 0.5:
            results['errors'].append(
                f"CRITICAL: Only {100*uniqueness_ratio:.1f}% unique claims! "
                "High risk of data leakage and memorization."
            )
        elif uniqueness_ratio < 0.75:
            results['warnings'].append(
                f"Low uniqueness ({100*uniqueness_ratio:.1f}%). "
                "Dataset may have excessive resampling."
            )
        else:
            results['passed_checks'].append("Uniqueness ratio acceptable (>75%)")
        
        # 3. Duplicate Analysis
        self._log("\n3. DUPLICATE DETECTION", "HEADER")
        
        duplicate_count = df[text_col].duplicated().sum()
        self._log(f"Exact duplicates: {duplicate_count:,}")
        
        if duplicate_count > 0:
            dup_ratio = duplicate_count / len(df)
            if dup_ratio > 0.1:
                results['errors'].append(
                    f"{100*dup_ratio:.1f}% of dataset is duplicated content!"
                )
            else:
                results['warnings'].append(
                    f"{duplicate_count} duplicates found ({100*dup_ratio:.2f}%)"
                )
        else:
            results['passed_checks'].append("No exact duplicates detected")
        
        # 4. Class Balance
        self._log("\n4. CLASS BALANCE", "HEADER")
        
        class_counts = df[label_col].value_counts()
        self._log(f"Class distribution:\n{class_counts.to_string()}")
        
        results['class_distribution'] = class_counts.to_dict()
        
        # Check balance
        if len(class_counts) > 0:
            min_class = class_counts.min()
            max_class = class_counts.max()
            ratio = min_class / max_class
            results['class_balance_ratio'] = ratio
            
            self._log(f"Balance ratio: {100*ratio:.1f}%")
            
            if ratio < 0.4:
                results['warnings'].append(
                    f"Severely imbalanced dataset ({100*ratio:.1f}% ratio). "
                    "Consider stratified sampling."
                )
            elif ratio < 0.8:
                results['warnings'].append(
                    f"Imbalanced classes ({100*ratio:.1f}% ratio). "
                    "May affect model training."
                )
            else:
                results['passed_checks'].append("Classes well-balanced")
        
        # 5. Missing Values
        self._log("\n5. MISSING VALUES", "HEADER")
        
        missing = df[[text_col, label_col]].isnull().sum()
        total_missing = missing.sum()
        
        self._log(f"Missing {text_col}: {missing.get(text_col, 0):,}")
        self._log(f"Missing {label_col}: {missing.get(label_col, 0):,}")
        
        if total_missing > 0:
            results['warnings'].append(f"{total_missing} total missing values")
        else:
            results['passed_checks'].append("No missing values in critical columns")
        
        results['missing_values'] = total_missing
        
        # 6. Text Length Analysis
        self._log("\n6. TEXT LENGTH STATISTICS", "HEADER")
        
        text_lengths = df[text_col].astype(str).str.len()
        
        self._log(f"Mean length: {text_lengths.mean():.0f} chars")
        self._log(f"Min length: {text_lengths.min():,} chars")
        self._log(f"Max length: {text_lengths.max():,} chars")
        self._log(f"Median length: {text_lengths.median():.0f} chars")
        
        results['text_stats'] = {
            'mean_length': text_lengths.mean(),
            'min_length': text_lengths.min(),
            'max_length': text_lengths.max(),
            'median_length': text_lengths.median()
        }
        
        # 7. Summary
        self._log("\n" + "="*70, "HEADER")
        self._log("VALIDATION SUMMARY", "HEADER")
        self._log("="*70, "HEADER")
        
        # Passed checks
        if results['passed_checks']:
            self._log(f"\n✓ Passed ({len(results['passed_checks'])}):")
            for check in results['passed_checks']:
                self._log(f"  • {check}", "SUCCESS")
        
        # Warnings
        if results['warnings']:
            self._log(f"\n⚠️  Warnings ({len(results['warnings'])}):")
            for warning in results['warnings']:
                self._log(f"  • {warning}", "WARNING")
        
        # Errors
        if results['errors']:
            self._log(f"\n❌ Errors ({len(results['errors'])}):")
            for error in results['errors']:
                self._log(f"  • {error}", "ERROR")
        
        # Quality score
        quality_score = 100
        quality_score -= len(results['warnings']) * 5
        quality_score -= len(results['errors']) * 20
        quality_score = max(0, min(100, quality_score))
        
        results['quality_score'] = quality_score
        
        self._log(f"\nOVERALL QUALITY SCORE: {quality_score}/100")
        
        if quality_score >= 80:
            self._log("✓ Dataset quality: GOOD", "SUCCESS")
        elif quality_score >= 60:
            self._log("⚠️  Dataset quality: ACCEPTABLE", "WARNING")
        else:
            self._log("❌ Dataset quality: POOR", "ERROR")
        
        return results
    
    def validate_splits(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        text_col: str = 'claim'
    ) -> dict:
        """
        Validate train/val/test split separation.
        
        Check for:
        - Data leakage between splits
        - Proper stratification
        - Size balance
        """
        
        self._log("\n" + "="*70, "HEADER")
        self._log("TRAIN/VAL/TEST SPLIT VALIDATION", "HEADER")
        self._log("="*70, "HEADER")
        
        # Load splits
        try:
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            df_test = pd.read_csv(test_path)
        except Exception as e:
            self._log(f"Failed to load splits: {e}", "ERROR")
            return {}
        
        results = {
            'train_size': len(df_train),
            'val_size': len(df_val),
            'test_size': len(df_test),
            'leakage_detected': False
        }
        
        # 1. Size Analysis
        self._log("\n1. SPLIT SIZES", "HEADER")
        total = len(df_train) + len(df_val) + len(df_test)
        
        self._log(f"Train: {len(df_train):,} ({100*len(df_train)/total:.1f}%)")
        self._log(f"Val:   {len(df_val):,} ({100*len(df_val)/total:.1f}%)")
        self._log(f"Test:  {len(df_test):,} ({100*len(df_test)/total:.1f}%)")
        self._log(f"Total: {total:,}")
        
        # 2. Overlap Detection
        self._log("\n2. LEAKAGE DETECTION", "HEADER")
        
        train_claims = set(df_train[text_col].astype(str))
        val_claims = set(df_val[text_col].astype(str))
        test_claims = set(df_test[text_col].astype(str))
        
        train_val_overlap = len(train_claims & val_claims)
        train_test_overlap = len(train_claims & test_claims)
        val_test_overlap = len(val_claims & test_claims)
        
        self._log(f"Train-Val overlap: {train_val_overlap}")
        self._log(f"Train-Test overlap: {train_test_overlap}")
        self._log(f"Val-Test overlap: {val_test_overlap}")
        
        total_overlap = train_val_overlap + train_test_overlap + val_test_overlap
        
        if total_overlap > 0:
            results['leakage_detected'] = True
            self._log(f"\n❌ DATA LEAKAGE DETECTED: {total_overlap} overlapping claims!", "ERROR")
            results['warnings'] = [f"Critical data leakage: {total_overlap} overlaps"]
        else:
            self._log("\n✓ No data leakage detected between splits", "SUCCESS")
            results['passed_checks'] = ["Proper data separation"]
        
        # 3. Class Balance per split
        self._log("\n3. CLASS BALANCE PER SPLIT", "HEADER")
        
        for split_name, split_df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
            balance = split_df['label'].value_counts()
            ratio = balance.min() / balance.max() if len(balance) > 1 else 1.0
            self._log(f"{split_name}: {dict(balance.to_dict())} (ratio: {100*ratio:.1f}%)")
        
        return results


def main():
    """Run validation on all datasets."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate dataset quality')
    parser.add_argument('--file', type=str, help='CSV file to validate')
    parser.add_argument('--train', type=str, help='Train CSV file')
    parser.add_argument('--val', type=str, help='Validation CSV file')
    parser.add_argument('--test', type=str, help='Test CSV file')
    parser.add_argument('--text', type=str, default='claim', help='Text column name')
    parser.add_argument('--label', type=str, default='label', help='Label column name')
    
    args = parser.parse_args()
    
    validator = DataQualityValidator(verbose=True)
    
    # Single file validation
    if args.file:
        validator.validate_dataset(args.file, args.text, args.label)
    
    # Split validation
    if args.train and args.val and args.test:
        print("\n\n")
        validator.validate_splits(args.train, args.val, args.test, args.text)
    
    # Default: check for common dataset paths
    if not args.file and not args.train:
        print("Checking for datasets in data/processed/...\n")
        
        data_dir = Path('data/processed')
        
        # Try to validate combined datasets
        for pattern in ['realworld_train.csv', 'synthetic_diverse_claims_10k.csv']:
            filepath = data_dir / pattern
            if filepath.exists():
                print(f"\n\nValidating {pattern}")
                validator.validate_dataset(str(filepath))
        
        # Try split validation
        train_path = data_dir / 'realworld_train.csv'
        val_path = data_dir / 'realworld_val.csv'
        test_path = data_dir / 'realworld_test.csv'
        
        if all(p.exists() for p in [train_path, val_path, test_path]):
            validator.validate_splits(str(train_path), str(val_path), str(test_path))


if __name__ == '__main__':
    main()
