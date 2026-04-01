# Google Colab Troubleshooting Guide

## Issue: FileNotFoundError - Data Files Missing

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 
'/tmp/training/misinformation-at-scale/data/processed/fakenewsnet_train.csv'
```

### Root Cause
The git repository doesn't include data files (too large). They need to be generated in Colab before training.

### ✅ Solution: Use COLAB_COMPLETE.py

This script:
1. ✓ Installs dependencies
2. ✓ Clones repository  
3. ✓ Creates data directories
4. ✓ **Downloads & generates FakeNewsNet data** (NEW!)
5. ✓ Verifies all files exist
6. ✓ Checks GPU availability
7. ✓ Runs complete training

---

## How to Fix in Google Colab

### Step 1: Create New Cell
In Google Colab, click **+ Code** to add a new cell

### Step 2: Copy This Code

```python
print("Step 1/5: Installing dependencies...")
!pip install -q pandas numpy scikit-learn torch transformers tqdm

print("\nStep 2/5: Cloning repository...")
!rm -rf misinformation-at-scale 2>/dev/null
!git clone https://github.com/sanjaykshetri/misinformation-at-scale.git
%cd misinformation-at-scale

print("\nStep 3/5: Creating data directory...")
import os
os.makedirs('data/processed', exist_ok=True)

print("\nStep 4/5: Downloading FakeNewsNet data...")
exec(open('load_fakenewsnet.py').read())

print("\nStep 5/5: Verifying data files...")
data_files = [
    'data/processed/fakenewsnet_train.csv',
    'data/processed/fakenewsnet_val.csv',
    'data/processed/fakenewsnet_test.csv'
]
for f in data_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024*1024)
        print(f"✓ {f} ({size:.1f} MB)")
    else:
        print(f"❌ {f} NOT FOUND")

import torch
print(f"\n✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")

print("\n" + "="*80)
print("🚀 STARTING COMPLETE TRAINING")
print("="*80 + "\n")

exec(open('run_complete_training.py').read())
```

### Step 3: Run
Click ▶️ (or Shift+Enter)

### Step 4: Wait
Training should complete in 30-60 minutes

---

## What Happens During Execution

```
Step 1/5: Installing dependencies...
  ✓ pandas, numpy, scikit-learn, torch, transformers, tqdm

Step 2/5: Cloning repository...
  ✓ Cloned misinformation-at-scale repo

Step 3/5: Creating data directory...
  ✓ data/processed/ created

Step 4/5: Downloading FakeNewsNet data...
  • Downloading PolitiFact data (1.5 MB)
  • Downloading GossipCop data (40 MB)
  • Processing and combining datasets
  ✓ fakenewsnet_train.csv (1.7 MB - 16,235 samples)
  ✓ fakenewsnet_val.csv (360 KB - 3,479 samples)
  ✓ fakenewsnet_test.csv (361 KB - 3,480 samples)

Step 5/5: Verifying data files...
  ✓ fakenewsnet_train.csv (1.7 MB)
  ✓ fakenewsnet_val.csv (0.4 MB)
  ✓ fakenewsnet_test.csv (0.4 MB)
  ✓ GPU: Tesla T4

🚀 STARTING COMPLETE TRAINING
════════════════════════════════════════════════
PART 1: BASELINE MODEL (Logistic Regression + TF-IDF)
  ✓ Training complete: 84.28% accuracy

PART 2: DEEP LEARNING MODEL (DistilBERT)
  ✓ Training complete: 85-90% accuracy expected

✅ TRAINING COMPLETE!
```

---

## Key Points

1. **Data Download**: The `load_fakenewsnet.py` script downloads real data from GitHub
2. **No Git LFS Needed**: Data is generated, not stored in git
3. **All in One Cell**: Everything happens in sequence
4. **Takes 40-70 minutes**: Most time is GPU training
5. **GPU Recommended**: CPU training would take 8+ hours

---

## If Still Having Issues

### Issue: "Module not found"
```python
!pip install --upgrade pip
!pip install pandas numpy scikit-learn torch transformers tqdm
```

### Issue: "Download failed"
The FakeNewsNet data comes from GitHub. If download fails:
1. Check internet connection
2. Try again (sometimes transient network issues)
3. Check GitHub status: https://github.com/madhusudhanrajashankar/Fake-News-Detection

### Issue: "Out of memory"
If CUDA runs out of memory:
```python
# In load_fakenewsnet.py or run_complete_training.py
batch_size = 8  # Reduce from 16 or 32
```

### Issue: Timeout (12+ hours)
Google Colab sessions timeout after 12 hours. To continue:
1. Save models from first run
2. Start new session
3. Load saved models

---

## Expected Results

After training completes, you should see:

```
✓ BASELINE MODEL RESULTS:
  Train Accuracy:  0.8610
  Val Accuracy:    0.8350
  Test Accuracy:   0.8428 (84.28%)
  Test AUC:        0.8772

✓ DEEP LEARNING MODEL RESULTS:
  Test Accuracy:   0.88xx - 0.90xx
  Test AUC:        0.91xx
  Improvement:     +4-6% over baseline

✓ Models saved to models/
```

---

## Files Available in Your Repo

```
COLAB_COMPLETE.py       ← Use this one! (includes data download)
COLAB_SIMPLE.py         ← Alternative (requires pre-downloaded data)
COLAB_FIXED.py          ← Alternative (cleanup version)
GOOGLE_COLAB_GUIDE.md   ← Full guide with all options
```

---

## Quick Reference

| Problem | Solution |
|---------|----------|
| Data files missing | Use COLAB_COMPLETE.py |
| GPU not available | Check Runtime → Change runtime type → GPU |
| Memory error | Reduce batch_size |
| Timeout (12h) | Use GPU (faster), or split training |
| Network error | Retry - sometimes transient |

---

## Next Run (If Needed)

If you want to retrain or experiment:
1. Same code works again
2. Old data will be overwritten
3. No cleanup needed - code handles it

---

**All set!** Copy the code above into Google Colab and run. 🚀
