# Misinformation Detection at Scale
### Real-world misinformation detection using FakeNewsNet and transformer-based deep learning

---

## Overview

This project addresses the critical problem of misinformation detection at scale using real-world fact-checked data and modern deep learning techniques. The system detects false claims with realistic accuracy using professional fact-checking labels from PolitiFact and GossipCop.

Built with **23,194 real fact-checked articles** from the FakeNewsNet dataset, this project demonstrates:

- **Real-world misinformation detection** (84-86% accuracy)
- **Data quality validation** (100% unique samples, 0 data leakage)
- **Baseline + Deep Learning comparison** (Logistic Regression vs. DistilBERT)
- **Production-ready deployment** (tested on Google Colab with GPU)
- **Comprehensive documentation** (model cards, ethical considerations, deployment guides)

**Key Achievement:** Fixed data leakage issue (100% → 84-86% realistic accuracy) by replacing synthetic data with real FakeNewsNet articles.

---

## Research Question

Can we detect misinformation in real-world news using professional fact-checking labels and modern NLP techniques?

**Answer:** Yes. Using FakeNewsNet data, we achieve realistic 84-86% accuracy with excellent generalization (2.5% train-val gap).

---

## Dataset

This project uses **FakeNewsNet** - a comprehensive fact-checking dataset compiled from professional fact-checkers.

### FakeNewsNet Dataset

| Property | Value |
|----------|-------|
| **Total Samples** | 23,194 real-world articles |
| **Source A** | PolitiFact (political claims) |
| **Source B** | GossipCop (celebrity gossip) |
| **Labels** | Professional fact-checker verdicts (Real/Fake) |
| **Data Uniqueness** | 100% (no duplicates) |
| **Data Leakage** | 0 (verified in splits) |
| **Training Set** | 16,235 samples (70%) |
| **Validation Set** | 3,479 samples (15%) |
| **Test Set** | 3,480 samples (15%) |
| **Class Balance** | 3:1 Real/Fake ratio (realistic) |

### Key Advantages Over Alternatives

| Aspect | FakeNewsNet | Synthetic Data | Reddit Comments |
|--------|-------------|---|---|
| **Realism** | ✅ Real articles | ❌ Artificial patterns | ⚠️ Proxy labels |
| **Data Leakage** | ✅ Verified 0 | ❌ Often high | ❌ Community-based |
| **Label Quality** | ✅ Expert verified | ❌ Uncertain | ⚠️ Weak supervision |
| **Generalization** | ✅ 2.5% gap | ❌ Overfitting | ⚠️ Unknown |
| **Domain Coverage** | ✅ Politics + Celebrity | ❌ Demo only | ⚠️ Forum discussions |

### Data Characteristics

- Real fact-checked claims from professional platforms
- Article text, source URL, and expert-verified labels
- Authentic misinformation patterns from real-world sources
- Stratified splits with no overlap between train/val/test
- Ready for immediate model training without preprocessing

**Note:** Data files are auto-downloaded by `load_fakenewsnet.py` script during training.

---

## Labeling Strategy

This project uses **professional expert labels** from established fact-checking platforms:

- **PolitiFact:** Non-partisan political fact-checking since 2007
- **GossipCop:** Celebrity rumor verification
- **Label Definition:** Binary classification (Real/Fake)
- **Annotation Quality:** Vetted by professionals, not crowd-sourced

This approach provides **ground-truth labels** for misinformation detection, enabling reliable supervised learning with realistic accuracy expectations.

---

## Project Architecture

### 1. Data Ingestion

- Auto-download FakeNewsNet from GitHub (42 MB)
- Extract PolitiFact and GossipCop articles
- Verify data integrity and uniqueness
- Parse article text, sources, and labels

**Script:** `load_fakenewsnet.py`

### 2. Data Processing

- Deduplication by article_id (prevents data leakage)
- Stratified train/val/test splits (70/15/15)
- Text normalization and encoding
- No data leakage verification (0 overlaps confirmed)

### 3. Exploratory Data Analysis

- Class distribution analysis (Real vs. Fake ratio)
- Text length statistics
- Label distribution verification
- Data leakage detection
- Linguistic feature exploration

**Notebook:** `notebooks/02_exploratory_analysis.ipynb`

### 4. Baseline Model

**Architecture:** Logistic Regression + TF-IDF

- TF-IDF vectorization (5000 features, bigrams)
- Logistic regression classifier
- Hyperparameter tuning with L2 regularization
- Fast CPU-based inference

**Performance:**
- Test Accuracy: **84.28%**
- Precision: 81.23% (high confidence)
- Recall: 47.62% (moderate coverage)
- AUC: 0.8777

**Notebook:** `notebooks/03_baseline_modeling.ipynb`

### 5. Deep Learning Model

**Architecture:** DistilBERT (Fine-tuned Transformer)

- Pre-trained DistilBERT (base, uncased) - 66M parameters
- Custom classification head (768 → 2 classes)
- Fine-tuned on FakeNewsNet with Adam optimizer
- GPU-accelerated training (3 epochs, Tesla T4)

**Performance:**
- Test Accuracy: **85.75%** (+1.47% vs baseline)
- **Recall: 63.73%** (+16.11% vs baseline)
- F1 Score: 0.6892 (+14.81% vs baseline)
- AUC: 0.8816

**Notebook:** `notebooks/04_deep_learning_model.ipynb`

### 6. Model Comparison & Evaluation

- Baseline vs. Deep Learning comparison table
- Precision-recall trade-off analysis
- Use case recommendations (speed vs. accuracy)
- Ensemble strategy documentation
- Error analysis and failure modes

**Documentation:** `FINAL_RESULTS.md`, `MODEL_CARD.md`

---

## Repository Structure

```
misinformation-at-scale/
│
├── notebooks/
│   ├── 01_data_wrangling.ipynb              # Data loading & preprocessing
│   ├── 02_exploratory_analysis.ipynb        # EDA & distributions
│   ├── 03_baseline_modeling.ipynb           # LR + TF-IDF baseline
│   └── 04_deep_learning_model.ipynb         # DistilBERT & comparison
│
├── src/
│   ├── config.py                            # Configuration management
│   ├── clean.py                             # Data cleaning utilities
│   └── ingest.py                            # Legacy Reddit ingestion
│
├── models/
│   ├── baseline_lr_model.pkl                # Logistic Regression model
│   └── distilbert_model.pt                  # DistilBERT weights
│
├── data/
│   ├── raw/                                 # Original downloads
│   └── processed/                           # Cleaned splits
│       ├── fakenewsnet_train.csv           # 16,235 training samples
│       ├── fakenewsnet_val.csv             # 3,479 validation samples
│       └── fakenewsnet_test.csv            # 3,480 test samples
│
├── load_fakenewsnet.py                     # Download & preprocess FakeNewsNet
├── run_complete_training.py                # End-to-end training pipeline
├── COLAB_COMPLETE.py                       # Google Colab entry point
│
├── FINAL_RESULTS.md                        # Comprehensive results report
├── MODEL_CARD.md                           # Model documentation & ethics
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## Quick Start

### Local Development

**1. Clone and install:**

```bash
git clone https://github.com/sanjaykshetri/misinformation-at-scale.git
cd misinformation-at-scale
pip install -r requirements.txt
```

**2. Download and prepare data (auto-runs during training):**

```bash
python load_fakenewsnet.py
```

This creates:
- `data/processed/fakenewsnet_train.csv` (16,235 samples, 70%)
- `data/processed/fakenewsnet_val.csv` (3,479 samples, 15%)
- `data/processed/fakenewsnet_test.csv` (3,480 samples, 15%)

**3. Train both models:**

```bash
python run_complete_training.py
```

Output:
- Baseline model accuracy: **84.28%**
- DistilBERT accuracy: **85.75%**
- Training time: ~30 seconds (baseline) + ~25 seconds (DL on CPU)

### Google Colab (GPU Training - Recommended)

For faster deep learning training on Tesla T4 GPU:

**1. Open Colab:** https://colab.research.google.com

**2. Set GPU runtime:** Runtime → Change runtime type → GPU

**3. Paste and run this code:**

```python
# One cell - copy from COLAB_COMPLETE.py
!pip install -q pandas numpy scikit-learn torch transformers tqdm
!git clone https://github.com/sanjaykshetri/misinformation-at-scale.git
%cd misinformation-at-scale
exec(open('COLAB_COMPLETE.py').read())
```

Expected time: **45-60 minutes** (includes setup + both model training)

**Results saved to:** `/content/misinformation-at-scale/models/`

---

## Setup Details

### Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- Scikit-learn 1.3+
- Pandas 2.0+

See `requirements.txt` for exact versions.

### Data Setup

**FakeNewsNet is automatically downloaded** by `load_fakenewsnet.py`:

- Source: GitHub (misinformation-at-scale repository)
- Size: ~42 MB (auto-decompressed)
- Format: CSV with article text and labels
- Processing: Auto-deduplication and train/val/test split

No additional downloads needed.

### Manual Alternative

If you prefer other real-world datasets:

#### LIAR Dataset (political claims)
```bash
# Download from https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
# Extract to data/raw/ and update load_real_datasets.py
```

#### FEVER Dataset (fact verification)
```bash
# Download from https://fever.ai/
# Process with validate_data_quality.py
```

---

## Usage Guide

### Complete Pipeline

```bash
# Step 1: Download & process FakeNewsNet data
python load_fakenewsnet.py

# Step 2: Train baseline + deep learning models
python run_complete_training.py

# Step 3: View results
cat FINAL_RESULTS.md
```

### Notebook Exploration

Run in order:

1. **`01_data_wrangling.ipynb`** - Load and explore FakeNewsNet
2. **`02_exploratory_analysis.ipynb`** - Statistical distributions
3. **`03_baseline_modeling.ipynb`** - Train LR baseline (84.28%)
4. **`04_deep_learning_model.ipynb`** - Train DistilBERT (85.75%)

Each notebook is standalone but builds on previous analysis.

### Model Inference

**Using Baseline Model (Fast):**

```python
import pickle
import pandas as pd

# Load model
with open('models/baseline_lr_model.pkl', 'rb') as f:
    lr_model, vectorizer = pickle.load(f)

# Predict
text = "Your claim here"
features = vectorizer.transform([text])
pred = lr_model.predict(features)[0]
conf = lr_model.predict_proba(features)[0].max()
print(f"{'FAKE' if pred else 'REAL'} ({conf:.1%} confidence)")
```

**Using DistilBERT (Accurate):**

```python
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from src.models import DistilBertClassifier

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistilBertClassifier().to(device)
model.load_state_dict(torch.load('models/distilbert_model.pt'))
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Predict
text = "Your claim here"
encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

model.eval()
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    pred = torch.argmax(logits, dim=1).item()
    conf = torch.softmax(logits, dim=1)[0][pred].item()

print(f"{'FAKE' if pred else 'REAL'} ({conf:.1%} confidence)")
```

---

## Results

### Model Performance Comparison

| Component | Baseline | DistilBERT | Notes |
|-----------|----------|------------|-------|
| **Accuracy** | 84.28% | **85.75%** | +1.47% improvement |
| **Recall** | 47.62% | **63.73%** | +16.11% (catches more fakes!) |
| **Precision** | **81.23%** | 75.03% | Baseline more conservative |
| **F1 Score** | 0.6004 | **0.6892** | +14.81% better balance |
| **AUC-ROC** | 0.8777 | **0.8816** | Both excellent |
| **Inference Speed** | **100K/sec** | 1K/sec | Baseline much faster |
| **Training Time** | <1 sec | 25 min | DL needs GPU |
| **Memory** | **150 MB** | 300 MB | Baseline lightweight |

### Key Findings

✅ **Realistic Accuracy:** 84-86% (NOT fake 100%)  
✅ **Excellent Generalization:** 2.5% train-val gap  
✅ **Data Quality Verified:** 100% unique, 0 leakage  
✅ **Deep Learning Advantage:** +16% recall for comprehensive coverage  
✅ **Production Ready:** Both models tested and validated  

See `FINAL_RESULTS.md` for comprehensive analysis and use case recommendations.

---

## Documentation

### Key Files

- **[FINAL_RESULTS.md](FINAL_RESULTS.md)** - Complete results report with comparison
- **[MODEL_CARD.md](MODEL_CARD.md)** - Model documentation, ethics, limitations
- **[COLAB_TROUBLESHOOTING.md](COLAB_TROUBLESHOOTING.md)** - Colab deployment guide

### Articles & Analysis

- **Data Leakage Fix:** Original issue was 100% accuracy from resampled synthetic data
- **Solution:** Real 23,194 FakeNewsNet articles (100% unique, 0 leakage)
- **Result:** Realistic 84-86% accuracy with excellent generalization

---

## Ethical Considerations

This project uses professional fact-checking labels to train misinformation detection models. Key ethical principles:

### Intended Use
- Assist fact-checkers and journalists in identifying potential misinformation
- Support content moderation workflows
- Provide evidence-based misinformation detection
- Enable human-in-the-loop review processes

### Not Intended For
- Autonomous censorship without human oversight
- Determining individual credibility or trustworthiness
- Making binding decisions without human review
- Suppressing legitimate political or social discourse

### Limitations & Bias
- **Language:** English-only training data
- **Domain Bias:** PolitiFact (politics-heavy), GossipCop (celebrity focus)
- **Temporal:** Training data from 2016-2017 (may be outdated)
- **False Positives:** Satire, sarcasm, and context-dependent claims may be misclassified
- **Inherited Bias:** Models inherit the biases of professional fact-checkers

### Recommendations
- Always use with human review teams
- Flag low-confidence predictions (0.45-0.55) for manual verification
- Retrain periodically on new data
- Monitor for performance drift in production
- Disclose model limitations to users
- Consider across multiple fact-checking sources

See `MODEL_CARD.md` for detailed ethical analysis.

---

## Skills Demonstrated

* Real-world fact-checking dataset integration
* Data quality validation and leakage detection
* Baseline model design (TF-IDF + Logistic Regression)
* Transformer fine-tuning (DistilBERT)
* PyTorch model training and evaluation
* Deep learning on GPU (Google Colab)
* Precision-recall trade-off analysis
* Production-ready code deployment
* Comprehensive documentation and model cards
* Reproducible machine learning pipelines

---

## Contributions

Improvements welcome! Areas of interest:

- [ ] Cross-domain evaluation (Reddit, Twitter, TikTok)
- [ ] Multi-language support
- [ ] Explainability analysis (attention visualization, SHAP)
- [ ] Error analysis and failure modes
- [ ] Ensemble methods combining baseline + DL
- [ ] FastAPI/Flask deployment examples
- [ ] Web interface for predictions

---

## References

### Datasets
- **FakeNewsNet:** Shu et al. (2020) - [Paper](https://arxiv.org/abs/1905.08707)
- **PolitiFact:** [https://www.politifact.com](https://www.politifact.com)
- **GossipCop:** [https://www.gossipcop.com](https://www.gossipcop.com)

### Models
- **DistilBERT:** Sanh et al. (2019) - [Paper](https://arxiv.org/abs/1910.01108)
- **BERT:** Devlin et al. (2019) - [Paper](https://arxiv.org/abs/1810.04805)

### Resources
- [HuggingFace Models](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Scikit-learn Guide](https://scikit-learn.org/)

---

## Citation

If you use this project in research, please cite:

```bibtex
@software{chhetri2026misinformation,
  author = {Chhetri, Sanjay},
  title = {Misinformation Detection at Scale: Real-world NLP with FakeNewsNet},
  year = {2026},
  url = {https://github.com/sanjaykshetri/misinformation-at-scale}
}
```

---

## Contact & Support

- **GitHub Issues:** Report bugs or request features
- **Email:** sanjay@example.com
- **Discussion:** See GitHub Discussions tab

---

## Changelog

**v2.0 (April 2026)** - FakeNewsNet Integration
- Replaced synthetic data with 23,194 real FakeNewsNet articles
- Fixed data leakage (0 overlaps verified)
- Added DistilBERT deep learning model (85.75%)
- Google Colab GPU training support
- Comprehensive documentation and model cards

**v1.0 (Earlier)** - Initial Reddit-based approach
- Contained 100% accuracy issue (synthetic data leakage)
- Used Reddit comments with community-based weak supervision
- Identified as containing fundamental data quality problems

---

## License

MIT License - See LICENSE file for details.

You are free to use, modify, and distribute this code for academic and commercial purposes.
