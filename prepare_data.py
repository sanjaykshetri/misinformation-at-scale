"""
Data preparation utility script.

Helps with:
- Creating directory structure
- Decompressing downloaded files
- Verifying NDJSON format
- Quick data health checks
"""

import os
import sys
import json
import bz2
import gzip
import shutil
from pathlib import Path
from typing import Optional
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparer:
    """Utility class for data preparation."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"
    
    def create_directories(self) -> None:
        """Create necessary directory structure."""
        logger.info("Creating directories...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "cache").mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created {self.raw_dir}")
        logger.info(f"✓ Created {self.processed_dir}")
    
    def decompress_bz2(self, filepath: Path, output_path: Optional[Path] = None) -> Path:
        """Decompress a .bz2 file."""
        if output_path is None:
            output_path = filepath.with_suffix('')
        
        if output_path.exists():
            logger.warning(f"Output file already exists: {output_path}")
            return output_path
        
        logger.info(f"Decompressing {filepath.name}...")
        with bz2.open(filepath, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ {output_path.name} ({file_size_mb:.1f} MB)")
        
        return output_path
    
    def decompress_gz(self, filepath: Path, output_path: Optional[Path] = None) -> Path:
        """Decompress a .gz file."""
        if output_path is None:
            output_path = filepath.with_suffix('')
        
        if output_path.exists():
            logger.warning(f"Output file already exists: {output_path}")
            return output_path
        
        logger.info(f"Decompressing {filepath.name}...")
        with gzip.open(filepath, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ {output_path.name} ({file_size_mb:.1f} MB)")
        
        return output_path
    
    def decompress_all(self, directory: Optional[Path] = None, remove_source: bool = False) -> None:
        """Decompress all compressed files in directory."""
        if directory is None:
            directory = self.raw_dir
        
        logger.info(f"Decompressing files in {directory}...")
        
        # Handle .bz2 files
        for bz2_file in directory.glob('*.bz2'):
            try:
                self.decompress_bz2(bz2_file)
                if remove_source:
                    bz2_file.unlink()
                    logger.info(f"  Deleted source: {bz2_file.name}")
            except Exception as e:
                logger.error(f"Failed to decompress {bz2_file.name}: {e}")
        
        # Handle .gz files
        for gz_file in directory.glob('*.gz'):
            # Skip .tar.gz and .ndjson.gz patterns that shouldn't be decompressed
            if gz_file.suffix == '.gz' and gz_file.stem.endswith(('.ndjson', '.tar')):
                continue
            
            try:
                self.decompress_gz(gz_file)
                if remove_source:
                    gz_file.unlink()
                    logger.info(f"  Deleted source: {gz_file.name}")
            except Exception as e:
                logger.error(f"Failed to decompress {gz_file.name}: {e}")
    
    def verify_ndjson(self, filepath: Path, sample_size: int = 5) -> bool:
        """Verify NDJSON format by checking first N lines."""
        logger.info(f"Verifying {filepath.name}...")
        
        valid_count = 0
        invalid_count = 0
        
        try:
            # Check first N lines
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    
                    try:
                        json.loads(line)
                        valid_count += 1
                        logger.debug(f"  Line {i+1}: ✓")
                    except json.JSONDecodeError as e:
                        invalid_count += 1
                        logger.warning(f"  Line {i+1}: ✗ Invalid JSON")
            
            # Count total lines
            logger.info(f"Counting total lines...")
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = sum(1 for _ in f)
            
            # File size
            file_size_gb = filepath.stat().st_size / (1024 ** 3)
            
            logger.info(f"✓ {filepath.name}")
            logger.info(f"  Total lines: {total_lines:,}")
            logger.info(f"  File size: {file_size_gb:.2f} GB")
            logger.info(f"  Sample: {valid_count}/{sample_size} valid lines")
            
            return invalid_count == 0
        
        except Exception as e:
            logger.error(f"Error verifying {filepath.name}: {e}")
            return False
    
    def verify_all_ndjson(self, directory: Optional[Path] = None) -> None:
        """Verify all NDJSON files in directory."""
        if directory is None:
            directory = self.raw_dir
        
        logger.info(f"Verifying NDJSON files in {directory}...\n")
        
        ndjson_files = list(directory.glob('*.ndjson')) + list(directory.glob('*.ndjson.gz'))
        
        if not ndjson_files:
            logger.warning(f"No NDJSON files found in {directory}")
            return
        
        for ndjson_file in sorted(ndjson_files):
            self.verify_ndjson(ndjson_file)
            logger.info("")
    
    def get_data_stats(self) -> dict:
        """Get statistics about raw data."""
        stats = {
            'raw_dir': str(self.raw_dir),
            'files': {},
            'total_size_gb': 0,
            'total_lines': 0
        }
        
        for filepath in sorted(self.raw_dir.glob('*.ndjson')):
            size_gb = filepath.stat().st_size / (1024 ** 3)
            
            # Quick line count (can be slow for large files)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)
            except:
                lines = None
            
            stats['files'][filepath.name] = {
                'size_gb': round(size_gb, 2),
                'lines': lines
            }
            stats['total_size_gb'] += size_gb
            if lines:
                stats['total_lines'] += lines
        
        return stats
    
    def print_data_stats(self) -> None:
        """Print statistics about data."""
        stats = self.get_data_stats()
        
        logger.info("\n" + "="*70)
        logger.info("DATA STATISTICS")
        logger.info("="*70)
        logger.info(f"\nLocation: {stats['raw_dir']}\n")
        
        if not stats['files']:
            logger.info("No NDJSON files found.")
            return
        
        logger.info("Files:")
        for filename, file_stats in stats['files'].items():
            size = file_stats['size_gb']
            lines = file_stats['lines']
            if lines:
                logger.info(f"  {filename:40} {size:8.2f} GB  {lines:15,} lines")
            else:
                logger.info(f"  {filename:40} {size:8.2f} GB  (not counted)")
        
        logger.info(f"\nTotal: {stats['total_size_gb']:.2f} GB  {stats['total_lines']:,} lines")
        logger.info("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Data preparation utility for Reddit misinformation project'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Create directory structure'
    )
    parser.add_argument(
        '--decompress',
        action='store_true',
        help='Decompress all .bz2 and .gz files'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify NDJSON files'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print data statistics'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all steps'
    )
    parser.add_argument(
        '--base-dir',
        default='.',
        help='Base directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    preparer = DataPreparer(args.base_dir)
    
    # Run requested operations
    if args.setup or args.all:
        preparer.create_directories()
    
    if args.decompress or args.all:
        preparer.decompress_all()
    
    if args.verify or args.all:
        preparer.verify_all_ndjson()
    
    if args.stats or args.all:
        preparer.print_data_stats()
    
    # If no args, print help and stats
    if not any([args.setup, args.decompress, args.verify, args.stats, args.all]):
        parser.print_help()
        print("\nCurrent data status:")
        preparer.print_data_stats()


if __name__ == '__main__':
    main()
