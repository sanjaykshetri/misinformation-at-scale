"""
PySpark pipeline for ingesting and cleaning large-scale Reddit comment datasets.

This module handles:
- Reading compressed NDJSON Reddit dumps
- Schema normalization
- Filtering by subreddit and time window
- Writing cleaned data to Parquet format for downstream analysis
"""

import os
import gzip
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, 
    DoubleType, IntegerType, BooleanType
)
from pyspark.sql.functions import (
    col, from_unixtime, to_timestamp, 
    lower, trim, coalesce, when, lit
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditDataIngestion:
    """
    Handles ingestion and preprocessing of Reddit comment datasets using PySpark.
    """
    
    def __init__(self, spark_session: Optional[SparkSession] = None):
        """
        Initialize RedditDataIngestion pipeline.
        
        Args:
            spark_session: Optional Spark session. If None, creates a new one.
        """
        if spark_session is None:
            self.spark = SparkSession.builder \
                .appName("RedditDataIngestion") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "8g") \
                .getOrCreate()
        else:
            self.spark = spark_session
        
        logger.info(f"Spark session initialized: {self.spark.version}")
    
    @staticmethod
    def get_reddit_schema() -> StructType:
        """
        Define the expected schema for Reddit comment data.
        
        Returns:
            StructType: Spark schema for Reddit comments
        """
        return StructType([
            StructField("author", StringType(), True),
            StructField("author_flair_text", StringType(), True),
            StructField("body", StringType(), True),
            StructField("created_utc", LongType(), True),
            StructField("score", IntegerType(), True),
            StructField("subreddit", StringType(), True),
            StructField("subreddit_id", StringType(), True),
            StructField("id", StringType(), True),
            StructField("parent_id", StringType(), True),
            StructField("link_id", StringType(), True),
            StructField("num_comments", IntegerType(), True),
            StructField("is_submitter", BooleanType(), True),
            StructField("edited", LongType(), True),
        ])
    
    def read_compressed_ndjson(self, filepath: str) -> DataFrame:
        """
        Read compressed NDJSON file and infer schema.
        
        Args:
            filepath: Path to compressed .ndjson.gz file
            
        Returns:
            DataFrame: Raw Reddit data as Spark DataFrame
        """
        logger.info(f"Reading compressed NDJSON from: {filepath}")
        
        try:
            df = self.spark.read \
                .option("mode", "PERMISSIVE") \
                .option("encoding", "utf-8") \
                .json(filepath)
            
            logger.info(f"Successfully read {df.count()} records")
            return df
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            raise
    
    def normalize_schema(self, df: DataFrame) -> DataFrame:
        """
        Normalize and clean the Reddit data schema.
        
        Args:
            df: Raw DataFrame from read_compressed_ndjson
            
        Returns:
            DataFrame: Normalized DataFrame with consistent schema
        """
        logger.info("Normalizing schema...")
        
        normalized = df.select(
            coalesce(col("author"), lit("[deleted]")).alias("author"),
            coalesce(col("subreddit"), lit("unknown")).alias("subreddit"),
            trim(coalesce(col("body"), lit(""))).alias("body"),
            col("created_utc").cast(LongType()).alias("created_utc"),
            col("score").cast(IntegerType()).alias("score"),
            col("id").alias("comment_id"),
            col("parent_id").alias("parent_id"),
            col("link_id").alias("link_id"),
            col("is_submitter").cast(BooleanType()).alias("is_submitter"),
        )
        
        # Convert Unix timestamp to readable datetime
        normalized = normalized.withColumn(
            "created_at",
            from_unixtime(col("created_utc")).cast("timestamp")
        )
        
        return normalized
    
    def filter_by_subreddit(
        self, 
        df: DataFrame, 
        subreddits: List[str]
    ) -> DataFrame:
        """
        Filter DataFrame to include only specified subreddits.
        
        Args:
            df: Input DataFrame
            subreddits: List of subreddit names to include
            
        Returns:
            DataFrame: Filtered to include only specified subreddits
        """
        logger.info(f"Filtering to subreddits: {subreddits}")
        
        filtered = df.filter(
            col("subreddit").isin([s.lower() for s in subreddits])
        )
        
        logger.info(f"Retained {filtered.count()} records after subreddit filter")
        return filtered
    
    def filter_by_time_window(
        self, 
        df: DataFrame, 
        start_date: str, 
        end_date: str
    ) -> DataFrame:
        """
        Filter DataFrame by time window (YYYY-MM-DD format).
        
        Args:
            df: Input DataFrame
            start_date: Start date as string (YYYY-MM-DD)
            end_date: End date as string (YYYY-MM-DD)
            
        Returns:
            DataFrame: Filtered to time window
        """
        logger.info(f"Filtering to time window: {start_date} to {end_date}")
        
        filtered = df.filter(
            (col("created_at") >= start_date) &
            (col("created_at") <= end_date)
        )
        
        logger.info(f"Retained {filtered.count()} records after time filter")
        return filtered
    
    def remove_deleted_comments(self, df: DataFrame) -> DataFrame:
        """
        Remove deleted or removed comments.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: With deleted comments removed
        """
        logger.info("Removing deleted/removed comments...")
        
        cleaned = df.filter(
            (col("body") != "[deleted]") &
            (col("body") != "[removed]") &
            (col("body") != "") &
            (col("body").isNotNull())
        )
        
        removed_count = df.count() - cleaned.count()
        logger.info(f"Removed {removed_count} deleted/empty comments")
        
        return cleaned
    
    def write_to_parquet(
        self, 
        df: DataFrame, 
        output_path: str, 
        mode: str = "overwrite"
    ) -> None:
        """
        Write DataFrame to Parquet format (partitioned by subreddit).
        
        Args:
            df: DataFrame to write
            output_path: Output directory path
            mode: Write mode (overwrite, append, ignore, error)
        """
        logger.info(f"Writing to Parquet: {output_path}")
        
        df.write \
            .mode(mode) \
            .partitionBy("subreddit") \
            .parquet(output_path)
        
        logger.info(f"Successfully wrote data to {output_path}")
    
    def ingest_pipeline(
        self,
        input_path: str,
        output_path: str,
        subreddits: List[str],
        start_date: str,
        end_date: str,
    ) -> DataFrame:
        """
        Complete ingestion pipeline from raw data to cleaned Parquet.
        
        Args:
            input_path: Path to input compressed NDJSON file
            output_path: Path to output Parquet directory
            subreddits: List of subreddits to include
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame: Final cleaned and processed data
        """
        logger.info("=" * 60)
        logger.info("Starting Reddit Data Ingestion Pipeline")
        logger.info("=" * 60)
        
        # Read raw data
        df = self.read_compressed_ndjson(input_path)
        
        # Normalize schema
        df = self.normalize_schema(df)
        
        # Apply filters
        df = self.filter_by_subreddit(df, subreddits)
        df = self.filter_by_time_window(df, start_date, end_date)
        df = self.remove_deleted_comments(df)
        
        # Write to Parquet
        self.write_to_parquet(df, output_path)
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 60)
        
        return df


def main():
    """
    Main entry point for ingestion pipeline.
    
    Reads configuration from environment or defaults.
    """
    # Configuration (typically read from config/settings.yaml)
    INPUT_PATH = os.getenv("REDDIT_INPUT_PATH", "data/raw/reddit_comments.ndjson.gz")
    OUTPUT_PATH = os.getenv("REDDIT_OUTPUT_PATH", "data/processed/reddit_comments_clean")
    SUBREDDITS = ["conspiracy", "theDonald", "science", "askscience"]  # Example subreddits
    START_DATE = "2020-01-01"
    END_DATE = "2020-12-31"
    
    # Initialize pipeline
    ingestion = RedditDataIngestion()
    
    # Run pipeline
    df = ingestion.ingest_pipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        subreddits=SUBREDDITS,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    
    # Show sample results
    logger.info("\nSample of processed data:")
    df.select("author", "subreddit", "body", "created_at", "score").show(5, truncate=False)
    
    # Summary statistics
    logger.info("\nProcessed Data Summary:")
    logger.info(f"Total records: {df.count()}")
    logger.info(f"Subreddit distribution:\n")
    df.groupBy("subreddit").count().show()


if __name__ == "__main__":
    main()
