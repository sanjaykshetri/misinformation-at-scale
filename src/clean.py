"""
Data cleaning and preprocessing utilities.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, length, lower, regexp_replace, 
    when, lit
)
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Utilities for cleaning and preprocessing Reddit comment data."""
    
    @staticmethod
    def filter_by_length(
        df: DataFrame, 
        min_length: int = 5, 
        max_length: int = 5000
    ) -> DataFrame:
        """
        Filter comments by text length.
        
        Args:
            df: Input DataFrame
            min_length: Minimum comment length
            max_length: Maximum comment length
            
        Returns:
            DataFrame: Filtered by length constraints
        """
        logger.info(f"Filtering by length: {min_length}-{max_length} chars")
        
        filtered = df.filter(
            (length(col("body")) >= min_length) &
            (length(col("body")) <= max_length)
        )
        
        removed = df.count() - filtered.count()
        logger.info(f"Removed {removed} comments outside length range")
        
        return filtered
    
    @staticmethod
    def filter_by_score(df: DataFrame, min_score: int = -5) -> DataFrame:
        """
        Filter comments by score threshold.
        
        Args:
            df: Input DataFrame
            min_score: Minimum comment score
            
        Returns:
            DataFrame: Filtered by score
        """
        logger.info(f"Filtering by minimum score: {min_score}")
        
        filtered = df.filter(col("score") >= min_score)
        
        removed = df.count() - filtered.count()
        logger.info(f"Removed {removed} low-score comments")
        
        return filtered
    
    @staticmethod
    def remove_urls(df: DataFrame) -> DataFrame:
        """
        Remove URLs from comment text.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: With URLs removed
        """
        logger.info("Removing URLs from comments")
        
        cleaned = df.withColumn(
            "body",
            regexp_replace(col("body"), r"http\S+|www\S+", "")
        )
        
        return cleaned
    
    @staticmethod
    def create_class_label(
        df: DataFrame,
        misinformation_subreddits: list,
        control_subreddits: list
    ) -> DataFrame:
        """
        Create binary label (misinformation=1, control=0).
        
        Args:
            df: Input DataFrame
            misinformation_subreddits: List of misinformation community names
            control_subreddits: List of control community names
            
        Returns:
            DataFrame: With 'label' column added
        """
        logger.info("Creating class labels from subreddit affiliation")
        
        labeled = df.withColumn(
            "label",
            when(
                col("subreddit").isin([s.lower() for s in misinformation_subreddits]),
                lit(1)
            ).when(
                col("subreddit").isin([s.lower() for s in control_subreddits]),
                lit(0)
            ).otherwise(lit(-1))  # Exclude other communities
        )
        
        labeled = labeled.filter(col("label") >= 0)
        
        logger.info("Labels created and unmatched communities filtered")
        return labeled
