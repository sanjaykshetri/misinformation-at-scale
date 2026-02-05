# BigQuery Setup Guide

Quick reference for getting Reddit data from Google Cloud BigQuery.

---

## ⚡ Quick Setup (10 minutes)

### 1. Create Google Cloud Account

- Visit: https://cloud.google.com/free
- Click **"Start free"**
- Sign in with Google account (or create one)
- Free tier includes: **1TB/month** query allowance
- **No credit card required** for free tier

### 2. Create Project

1. Go to: https://console.cloud.google.com/
2. At top, click **"Select a Project"**
3. Click **"NEW PROJECT"**
4. Name: `misinformation-analysis`
5. Click **"Create"**

### 3. Enable BigQuery API

1. Search for **"BigQuery API"** in search bar
2. Click **"BigQuery API"** from results
3. Click **"Enable"**

### 4. Open BigQuery Console

- Visit: https://console.cloud.google.com/bigquery

---

## 📊 Query Reddit Data

### ⚠️ Note: Public Reddit Dataset Access

The `bigquery-public-data.reddit.comments` dataset has restricted access. Here are your **best alternatives**:

---

## ✅ Alternative 1: Use Pushshift API (Recommended)

Free, no authentication needed, works immediately:

```bash
# Simple download script
python download_reddit.py
```

Create `download_reddit.py`:

```python
import requests
import json
import time
from datetime import datetime, timedelta

def get_reddit_comments(subreddit, limit=10000):
    """Download comments from Pushshift API."""
    base_url = "https://api.pushshift.io/reddit/comment/search"
    comments = []
    
    # Get comments from 2020
    start_date = int(datetime(2020, 1, 1).timestamp())
    end_date = int(datetime(2020, 12, 31).timestamp())
    
    params = {
        'subreddit': subreddit,
        'after': start_date,
        'before': end_date,
        'size': 1000,
        'sort': 'asc'
    }
    
    print(f"Downloading {subreddit} comments...")
    while len(comments) < limit:
        try:
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            
            if not data.get('data'):
                break
            
            comments.extend(data['data'])
            params['after'] = data['data'][-1]['created_utc']
            
            print(f"  Downloaded {len(comments):,} comments...")
            time.sleep(1)  # Be respectful to API
            
        except Exception as e:
            print(f"  Error: {e}")
            break
    
    return comments[:limit]

# Download from multiple subreddits
all_comments = []
for sub in ['conspiracy', 'theDonald', 'science', 'askscience']:
    all_comments.extend(get_reddit_comments(sub, limit=2500))

# Save to CSV
import pandas as pd
df = pd.DataFrame(all_comments)
df.to_csv('data/raw/reddit_comments_2020.csv', index=False)
print(f"✓ Saved {len(df):,} comments to data/raw/reddit_comments_2020.csv")
```

**Advantages:**
- ✅ Free, no account needed
- ✅ Direct access to Reddit data
- ✅ Fast and simple
- ✅ Works immediately

---

## ✅ Alternative 2: Hugging Face Datasets

Pre-processed Reddit datasets available:

```python
from datasets import load_dataset

# Load Reddit dataset
data = load_dataset('reddit', split='train')

# Save to CSV
import pandas as pd
df = pd.DataFrame(data[:10000])
df.to_csv('data/raw/reddit_comments_2020.csv', index=False)
```

---

## ✅ Alternative 3: BigQuery (If You Have Access)

If you have a private BigQuery dataset or the public dataset becomes available:

```sql
SELECT
    author,
    body,
    created_utc,
    score,
    subreddit,
    id
FROM
    `bigquery-public-data.reddit.comments`
WHERE
    EXTRACT(YEAR FROM TIMESTAMP_MILLIS(created_utc*1000)) = 2020
    AND subreddit IN ('conspiracy', 'theDonald', 'NoNewNormal', 'science', 'askscience', 'news')
    AND body NOT IN ('[deleted]', '[removed]')
    AND LENGTH(body) > 5
LIMIT 100000
```

If you get "access denied", this dataset is restricted in your region/account.

---

## 🎯 Recommended: Use Pushshift (Option 1)

**Best path forward:**

1. Install dependencies:
   ```bash
   pip install requests pandas
   ```

2. Create and run `download_reddit.py` (code above)

3. CSV automatically saves to `data/raw/reddit_comments_2020.csv`

4. Run notebooks:
   ```bash
   jupyter notebook notebooks/01_data_wrangling.ipynb
   ```

**Time:** ~5-10 minutes to download 10k comments

---

## 💾 Export Results

### To CSV (Easiest)

1. Query runs → Results show at bottom
2. Click **"SAVE RESULTS"** button
3. Click **"CSV"** from dropdown
4. Choose filename: `reddit_comments_2020.csv`
5. File downloads automatically
6. Move to `data/raw/` directory

### To Google Drive (Alternative)

1. Results → **"SAVE RESULTS"**
2. Select **"Google Drive"**
3. Choose folder
4. File syncs to Drive
5. Download CSV locally
6. Move to `data/raw/`

---

## 📈 Scaling Up (Optional)

### For Pushshift: Get More Comments

Modify `download_reddit.py` to increase limits:

```python
# Change limit parameter
all_comments = []
for sub in ['conspiracy', 'theDonald', 'science', 'askscience', 'NoNewNormal']:
    all_comments.extend(get_reddit_comments(sub, limit=50000))  # Increased to 50k per subreddit
```

**Note:** Larger downloads take longer (respect Pushshift rate limits with `time.sleep`)

### For BigQuery: Query 1 Million Records

If BigQuery dataset becomes available:

```sql
-- Query 1 million records
SELECT
    author,
    body,
    created_utc,
    score,
    subreddit,
    id
FROM
    `bigquery-public-data.reddit.comments`
WHERE
    EXTRACT(YEAR FROM TIMESTAMP_MILLIS(created_utc*1000)) = 2020
    AND subreddit IN ('conspiracy', 'theDonald', 'NoNewNormal', 'science', 'askscience', 'news')
    AND body NOT IN ('[deleted]', '[removed]')
    AND LENGTH(body) > 5
LIMIT 1000000  -- Increased to 1 million
```

**Cost:** Still free (under 1TB limit)

---

## 🔧 Convert CSV to NDJSON (Optional)

If you want NDJSON format instead of CSV:

```python
# convert_to_ndjson.py
import pandas as pd
import json
import sys

def csv_to_ndjson(csv_path, output_path=None):
    """Convert BigQuery CSV to NDJSON format."""
    if output_path is None:
        output_path = csv_path.replace('.csv', '.ndjson')
    
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Writing {output_path}...")
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    
    print(f"✓ Converted {len(df):,} records")
    print(f"  Original: {csv_path}")
    print(f"  Output: {output_path}")

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/reddit_comments_2020.csv'
    csv_to_ndjson(csv_file)
```

Run with:
```bash
python convert_to_ndjson.py data/raw/reddit_comments_2020.csv
```

---

## ✅ Next Steps

### For Pushshift (Recommended)

1. **Create** `download_reddit.py` (code in Alternative 1 above)
2. **Run:**
   ```bash
   python download_reddit.py
   ```
3. **Check:** CSV appears in `data/raw/reddit_comments_2020.csv`
4. **Run notebooks:**
   ```bash
   jupyter notebook notebooks/01_data_wrangling.ipynb
   ```

### For BigQuery (If Access Available)

1. **Download CSV** from BigQuery → `data/raw/`
2. **Run notebooks:**
   ```bash
   jupyter notebook notebooks/01_data_wrangling.ipynb
   ```

### For Hugging Face

1. **Run:**
   ```python
   from datasets import load_dataset
   data = load_dataset('reddit', split='train')
   ```
2. **Save to CSV**
3. **Run notebooks**

---

**Fastest path: Pushshift script above** ⚡

---

## 🆘 Troubleshooting

### "Access Denied: Table bigquery-public-data:reddit.comments"

**This is a known issue.** The BigQuery public Reddit dataset has restricted access.

**Solutions:**
1. ✅ **Use Pushshift API instead** (recommended)
   - No authentication needed
   - Works immediately
   - See Alternative 1 above

2. Use Hugging Face Datasets
3. Download from Academic Torrents
4. Wait for BigQuery dataset to become available

### Pushshift API Issues

**"Connection timeout"**
- Pushshift API can be slow sometimes
- Add `time.sleep(2)` between requests
- Retry later

**"No data returned"**
- Check subreddit names are spelled correctly
- Try a different time range
- Check Pushshift status: https://api.pushshift.io/

### "CSV download not showing"

- Check browser downloads folder
- Try running download_reddit.py directly
- Verify `data/raw/` directory exists

### "Not enough results"

- Increase LIMIT in script
- Add more subreddits
- Extend date range (add 2019 data)

---

## 📊 Data Sources Comparison

| Source | Access | Speed | Quality | Ease |
|--------|--------|-------|---------|------|
| Pushshift API | ✅ Free, open | ~10 min | Good | ✅ Easy |
| BigQuery | ⚠️ Restricted | 2 min | Excellent | ⚠️ Auth needed |
| Hugging Face | ✅ Free | 5 min | Good | ✅ Easy |
| Academic Torrents | ⚠️ Large download | Hours | Excellent | Hard |

---

## 📝 Free Tier Limits

- **Query:** 1TB/month (this project uses <10GB)
- **Storage:** 10GB free
- **Duration:** No time limit (account lives as long as you use it)

---

## 💡 Pro Tips

1. **Test with small queries first** (LIMIT 1000)
2. **Add filters early** (subreddit, date) to reduce query size
3. **Monitor usage:** Check "Executed queries" to see data scanned
4. **Cache results:** BigQuery stores query results for 24 hours
5. **Use specific columns:** Don't SELECT * (shown in example above)

---

## 📚 Resources

- [BigQuery Public Datasets](https://console.cloud.google.com/marketplace/browse?filter=solution-type:dataset)
- [BigQuery SQL Reference](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax)
- [Reddit Data Documentation](https://www.reddit.com/r/bigquery/comments/3cej2b/)

---

**Total time to get data: ~15 minutes** ⚡

After this, your notebooks will run with real Reddit data!
