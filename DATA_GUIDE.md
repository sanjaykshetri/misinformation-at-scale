# Reddit Data Guide: Sourcing & Preparation

## Overview

This project uses publicly available Reddit comment archives. This guide explains how to obtain, prepare, and structure the data for the pipeline.

---

## 1. Data Sources

### Quick Start: Synthetic Data (Recommended for Immediate Testing)

**Best for:** Getting started immediately, validating pipeline logic, demos

- ✅ **No downloads needed**
- ✅ Runs in seconds
- ✅ Tests full ML pipeline end-to-end
- Notebook 01 has built-in fallback generator
- Perfect for Springboard review and recruiter demos

### Option A: BigQuery Public Dataset (Recommended for Real Data)

**Best for:** Working with actual Reddit data without large downloads

- **Google Cloud:** https://console.cloud.google.com/bigquery
- ✅ Free tier: 1TB/month queries (enough for this project)
- ✅ Pre-processed, ready-to-use format
- ✅ No decompression or format conversion needed
- ✅ Query exactly what you need from cloud
- Dataset: `bigquery-public-data.reddit.comments`
- Time to run: 2-3 hours setup (detailed instructions below)

### Option B: Academic Torrents (Alternative)

For downloading raw files locally (requires significant storage):

- **Academic Torrents:** https://academictorrents.com/
  - Search for "Reddit" or "RC_2020" (Reddit Comments 2020)
  - Files are in `.bz2` or `.gz` compressed format
  - Each file contains ~1 month of Reddit data as NDJSON
  - Storage needed: 25-50GB

### Option C: Other Sources

- **Pushshift Reddit Archive** (historical, some access restrictions)
- **Hugging Face Datasets** (pre-processed versions)
- **Custom sampling** (small local test files)

---

## 2. Data Format

Reddit comment files are typically **NDJSON** (one JSON object per line):

```json
{
  "author": "username",
  "body": "This is the comment text...",
  "created_utc": 1577836800,
  "score": 42,
  "subreddit": "conspiracy",
  "subreddit_id": "t5_xxxxx",
  "id": "abc123",
  "parent_id": "t1_xyz789",
  "link_id": "t3_aaa111",
  "is_submitter": false,
  "edited": false
}
```

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `author` | string | Username (or "[deleted]") |
| `body` | string | Comment text |
| `created_utc` | int | Unix timestamp |
| `score` | int | Upvotes - downvotes |
| `subreddit` | string | Community name |
| `id` | string | Comment ID |
| `parent_id` | string | Parent comment/submission ID |
| `is_submitter` | bool | Is author of original post |

---

## 3. Directory Structure

After downloading, organize files as follows:

```
misinformation-at-scale/
├── data/
│   ├── raw/                          # Raw downloaded files
│   │   ├── reddit_comments_2020_01.ndjson.gz
│   │   ├── reddit_comments_2020_02.ndjson.gz
│   │   └── ... (more monthly files)
│   │
│   └── processed/                    # Pipeline outputs
│       ├── reddit_comments_clean/    # After ingest.py
│       │   ├── _SUCCESS
│       │   ├── part-00000.parquet
│       │   └── ... (partitioned by subreddit)
│       │
│       └── reddit_comments_for_modeling/
│           └── ... (cleaned + labeled data)
│
├── config/settings.yaml
├── src/ingest.py
└── ... (other project files)
```

---

## 4. Getting Data: BigQuery Approach (Recommended)

### Step 1: Set Up Google Cloud Free Tier

1. Go to https://console.cloud.google.com/
2. Create a free Google Cloud account (if you don't have one)
3. Create a new project: "misinformation-analysis"
4. Enable BigQuery API
5. Go to BigQuery: https://console.cloud.google.com/bigquery

### Step 2: Query Reddit Data

In BigQuery Console, run this query:

```sql
-- Query 2020 Reddit data from misinformation subreddits
SELECT
    author,
    body,
    created_utc,
    score,
    subreddit,
    id,
    CAST(TIMESTAMP_MILLIS(created_utc*1000) AS DATE) as date
FROM
    `bigquery-public-data.reddit.comments`
WHERE
    -- Time window: All of 2020
    EXTRACT(YEAR FROM TIMESTAMP_MILLIS(created_utc*1000)) = 2020
    -- Misinformation subreddits
    AND subreddit IN ('conspiracy', 'theDonald', 'NoNewNormal')
    -- Remove deleted/removed comments
    AND body NOT IN ('[deleted]', '[removed]')
    -- Minimum comment length
    AND LENGTH(body) > 5
LIMIT 100000  -- Start with 100k, increase as needed
```

### Step 3: Export to CSV

1. Click **SAVE RESULTS** at bottom of query results
2. Choose **"Save to CSV"**
3. Name it: `reddit_comments_2020.csv`
4. Save to `data/raw/`

**Alternative: Export to Google Drive**
- Results → Save → Google Drive
- Download CSV to `data/raw/`

### Step 4: Scale Up (Optional)

Once you've tested with 100k records, increase the query:

```sql
-- For 1 million records
WHERE
    EXTRACT(YEAR FROM TIMESTAMP_MILLIS(created_utc*1000)) = 2020
    AND subreddit IN (
        'conspiracy', 'theDonald', 'NoNewNormal',  -- misinformation
        'science', 'askscience', 'news'             -- control
    )
    AND body NOT IN ('[deleted]', '[removed]')
    AND LENGTH(body) > 5
LIMIT 1000000
```

### Converting CSV to NDJSON (if needed)

If you want NDJSON format for compatibility with `ingest.py`:

```python
import pandas as pd
import json

# Read CSV from BigQuery
df = pd.read_csv('data/raw/reddit_comments_2020.csv')

# Convert to NDJSON
with open('data/raw/reddit_comments_2020.ndjson', 'w') as f:
    for _, row in df.iterrows():
        f.write(json.dumps(row.to_dict()) + '\n')

print(f"Saved {len(df):,} records to NDJSON")
```

---

## 4b. Alternative: Download from Academic Torrents (If Preferred)

### Step 1: Download Data

From **Academic Torrents**:

1. Visit https://academictorrents.com/
2. Search for "RC_2020" or "Reddit Comments 2020"
3. Download `.torrent` file
4. Use torrent client (qBittorrent, Transmission, etc.)
5. Extract to `data/raw/`

Example files to download:
- `RC_2020-01.bz2` (January 2020)
- `RC_2020-02.bz2` (February 2020)
- etc.

### Step 2: Decompress Files

```bash
# If .bz2 format
bunzip2 data/raw/RC_2020-01.bz2

# If .gz format
gunzip data/raw/RC_2020-01.gz

# Or use Python
import bz2
with bz2.open('data/raw/RC_2020-01.bz2', 'rt') as f:
    with open('data/raw/RC_2020-01.ndjson', 'w') as out:
        out.write(f.read())
```

### Step 3: Verify Format

```bash
# Check first line
head -1 data/raw/reddit_comments_2020_01.ndjson | python -m json.tool

# Count lines
wc -l data/raw/reddit_comments_2020_01.ndjson
```

---

## 5. Running the Ingestion Pipeline

Once data is in `data/raw/`:

```bash
# Install dependencies
pip install -r requirements.txt

# Run ingestion pipeline
python src/ingest.py
```

The pipeline will:
1. Read compressed NDJSON files
2. Normalize schema
3. Filter by specified subreddits and time window
4. Remove deleted/[removed] comments
5. Write clean data to `data/processed/reddit_comments_clean/` (Parquet format)

### Configuration in `config/settings.yaml`:

```yaml
ingestion:
  misinformation_subreddits:
    - conspiracy
    - theDonald
    - NoNewNormal
  
  control_subreddits:
    - science
    - askscience
    - news
  
  time_window:
    start_date: "2020-01-01"
    end_date: "2020-12-31"
```

---

## 6. Data Size & Performance

### Typical File Sizes

| Period | Compressed | Uncompressed | Comments |
|--------|-----------|--------------|----------|
| 1 month | 4-6 GB | 40-60 GB | 50-100M |
| 6 months | 25-30 GB | 250-300 GB | 300-600M |

### Storage Recommendations

- **Development:** 1-2 months of data (~5GB compressed)
- **Production:** 6-12 months (~25-50GB compressed)
- **Parquet output:** ~20-30% of uncompressed size

---

## 7. Quick Testing with Synthetic Data

**Don't have data yet? No problem!**

Notebook 01 automatically generates synthetic data if real files aren't found:

```python
# In 01_data_wrangling.ipynb
if not data_found:
    print("Generating synthetic data for testing...")
    data = generate_synthetic_reddit_data(n_samples=10000)
```

**This means:**
- ✅ Run all 4 notebooks immediately
- ✅ Full pipeline works end-to-end
- ✅ Show recruiters complete ML workflow
- ✅ Validate code before adding real data

**Then add real data later:**
1. Query BigQuery
2. Save to `data/raw/`
3. Re-run notebooks with real data
4. Compare results

### Sample Sizes for Testing

| Size | Time to Query | Time to Train | Best For |
|------|--------------|---------------|----------|
| 10k (synthetic) | Instant | 2 min | Immediate testing |
| 100k (BigQuery) | 30 sec | 10 min | Validation |
| 1M (BigQuery) | 2 min | 1 hour | Real results |
| 10M (Academic Torrents) | - | 4+ hours | Full analysis |

---

## 8. Data Privacy & Ethics

### Important Notes

- **Public data**: All Reddit data analyzed is from public comments
- **Anonymization**: We label by subreddit, not individual users
- **No personal info**: We don't store personally identifiable information
- **Academic use**: This is legitimate for academic research
- **Terms of Service**: Verify compliance with Reddit's ToS for your use case

### Citation

If publishing results, cite the dataset:
```
Reddit Comments Dataset, Academic Torrents, [Year]
https://academictorrents.com/
```

---

## 9. Recommended Workflow

### For Immediate Results (Today)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run notebooks with synthetic data
jupyter notebook notebooks/01_data_wrangling.ipynb

# This shows complete pipeline works without any data download
```

**Result:** Full ML pipeline demonstration in 2 hours

---

### For Real Data (This Week)

```bash
# 1. Query BigQuery (see Section 4)
#    This takes 2-3 hours (mostly setup time)

# 2. Download CSV from BigQuery to data/raw/

# 3. Run notebooks with real data
jupyter notebook notebooks/01_data_wrangling.ipynb
```

**Result:** Same pipeline but with actual Reddit data

---

### For Decompression (Academic Torrents Only)

If you're using `.bz2` files from Academic Torrents:

```python
# decompress_data.py
import os
import bz2
import shutil
from pathlib import Path

def decompress_all(raw_dir='data/raw'):
    """Decompress all .bz2 files in directory."""
    raw_path = Path(raw_dir)
    
    for bz2_file in raw_path.glob('*.bz2'):
        output_file = bz2_file.with_suffix('')
        
        if output_file.exists():
            print(f"Skipping {output_file} (already exists)")
            continue
        
        print(f"Decompressing {bz2_file.name}...")
        with bz2.open(bz2_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"✓ {output_file.name}")

if __name__ == '__main__':
    decompress_all()
```

Run with:
```bash
python decompress_data.py
```

---

## 10. Troubleshooting

### Issue: Out of Memory

**Solution:**
- Process files one at a time in `ingest.py`
- Increase Spark driver memory: `--driver-memory 8g`
- Reduce batch size in notebooks

### Issue: Slow Processing

**Solution:**
- Use Parquet format (faster than NDJSON)
- Increase number of Spark partitions
- Filter data early (by subreddit/date first)

### Issue: Missing Fields

**Solution:**
- Some fields may be missing in certain records
- `ingest.py` uses `coalesce()` to handle nulls
- Check schema validation in Notebook 01

### Issue: Imbalanced Classes

**Solution:**
- Use `class_weight='balanced'` in sklearn models (done ✓)
- Consider stratified k-fold CV (done ✓)
- Use focal loss in deep learning if needed

---

## 11. Running the Full Pipeline

### Path 1: Start Today (Synthetic Data)

```bash
# This works immediately
jupyter notebook notebooks/01_data_wrangling.ipynb

# Continue through all notebooks
jupyter notebook notebooks/02_exploratory_analysis.ipynb
jupyter notebook notebooks/03_baseline_modeling.ipynb
jupyter notebook notebooks/04_deep_learning_model.ipynb
```

**Timeline:** 2-3 hours (complete ML pipeline)

---

### Path 2: Add Real Data Later

1. **Query BigQuery** (following Section 4)
2. **Save CSV to** `data/raw/reddit_comments_2020.csv`
3. **Run notebooks again:**
   - `01_data_wrangling.ipynb` — Validate + EDA (now with real data)
   - `02_exploratory_analysis.ipynb` — Linguistic analysis
   - `03_baseline_modeling.ipynb` — TF-IDF + Logistic Regression
   - `04_deep_learning_model.ipynb` — DistilBERT fine-tuning

**Timeline:** 2-4 hours (real Reddit data results)

---

### Final Outputs

- ✅ Models saved in `distilbert_misinformation_classifier/` (HuggingFace format)
- ✅ Metrics and visualizations in notebook outputs
- ✅ Feature importance analysis (text & charts)
- ✅ Ready for deployment or further analysis

---

## Resources

- **Academic Torrents:** https://academictorrents.com/
- **Reddit Data Documentation:** https://www.reddit.com/r/pushshift/
- **PySpark SQL Docs:** https://spark.apache.org/docs/latest/sql-programming-guide.html
- **NDJSON Format:** http://ndjson.org/

---

## Questions?

Refer to:
- `config/settings.yaml` for pipeline configuration
- `src/ingest.py` for ingestion logic
- Notebook 01 for synthetic data generation
