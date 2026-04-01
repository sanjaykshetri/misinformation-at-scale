# Model Card: Misinformation Detection

## Model Details

### Model Description
- **Model Name:** Dual-Model Misinformation Detection System
- **Model Type:** Two complementary models for different use cases
- **Framework:** PyTorch + Transformers (DistilBERT), Scikit-learn (Baseline)
- **License:** MIT (See LICENSE file)
- **Model Date:** April 1, 2026

---

## Models Overview

### 1. Baseline Model: Logistic Regression + TF-IDF

**Purpose:** Fast, interpretable misinformation detection

**Architecture:**
- TF-IDF Vectorizer (5000 features, unigrams + bigrams)
- Logistic Regression classifier
- L2 regularization

**Performance:**
```
Test Accuracy:  84.28%
Precision:      81.23%
Recall:         47.62%
F1 Score:       0.6004
AUC-ROC:        0.8777
```

**Intended Use:**
- Real-time inference on edge devices
- Content priority scoring
- Explainability-focused applications
- Resource-constrained environments

**Limitations:**
- Lower recall (misses some misinformation)
- Requires manual feature engineering
- Cannot capture semantic nuances

---

### 2. Deep Learning Model: DistilBERT

**Purpose:** Comprehensive misinformation detection with semantic understanding

**Architecture:**
- DistilBERT (base, uncased) - 66M parameters
- Custom classification head (768 → 2 classes)
- Fine-tuned on FakeNewsNet dataset
- 3 epochs training on Tesla T4 GPU

**Performance:**
```
Test Accuracy:  85.75%
Precision:      75.03%
Recall:         63.73%
F1 Score:       0.6892
AUC-ROC:        0.8816
```

**Intended Use:**
- Batch processing (thousands of articles)
- Content moderation platforms
- Comprehensive fact-checking systems
- GPU-enabled production environments

**Strengths:**
- Better recall (+16% vs baseline)
- Learns semantic patterns
- Transfer learning benefit
- Can process longer texts

---

## Training Data

**Dataset:** FakeNewsNet
- **Source:** Real fact-checking platforms (PolitiFact + GossipCop)
- **Total Samples:** 23,194 real-world articles
- **Training Set:** 16,235 samples (70%)
- **Validation Set:** 3,479 samples (15%)
- **Test Set:** 3,480 samples (15%)

**Data Characteristics:**
- Real/Fake ratio: 3:1 (realistic distribution)
- Claim types: Politics, celebrity gossip, health
- Label source: Professional fact-checkers
- Data uniqueness: 100% (no duplicates)
- Data leakage: 0 (no overlaps between splits)

**Preprocessing:**
- Deduplication by article_id
- Stratified train/val/test splits
- Lowercasing + tokenization for BERT
- No synthetic data or resampling

---

## Evaluation Results

### Quantitative Analysis

| Metric | Baseline | DistilBERT | Winner |
|--------|----------|------------|--------|
| **Accuracy** | 84.28% | 85.75% | DistilBERT (+1.47%) |
| **Precision** | 81.23% | 75.03% | Baseline (+6.2%) |
| **Recall** | 47.62% | 63.73% | DistilBERT (+16.1%) |
| **F1 Score** | 0.6004 | 0.6892 | DistilBERT (+14.8%) |
| **AUC-ROC** | 0.8777 | 0.8816 | DistilBERT (+0.4%) |

### Analysis

**Accuracy:** DistilBERT provides modest improvement (+1.47%), but combined with better recall, offers better overall performance.

**Precision vs Recall Trade-off:**
- Baseline optimizes for precision (fewer false positives)
- DistilBERT optimizes for recall (catches more fakes)
- Both approaches have merit depending on application

**F1 Score:** DistilBERT's +14.8% improvement in F1 shows better balance between precision and recall.

---

## Model Comparison & Selection Guide

### Use Baseline Model When:
- ✅ Inference speed is critical (<1ms per prediction)
- ✅ Running on CPU or mobile devices
- ✅ High precision required (minimize false positives)
- ✅ Model interpretability is essential
- ✅ Low memory budget (<200 MB)

**Example:** Filtering user-reported content before sending to human reviewers

### Use DistilBERT When:
- ✅ Comprehensive coverage needed (catch more misinformation)
- ✅ GPU available for inference
- ✅ Batch processing (1000+ items)
- ✅ Better F1 score and recall matter more
- ✅ Budget allows 300 MB model size

**Example:** Content moderation for large platforms, automated fact-checking

### Ensemble Both Models:
- ✅ Maximize recall (flag anything either model suspects)
- ✅ Then use baseline precision for prioritization
- ✅ Best accuracy with reasonable latency

---

## Input/Output Specifications

### Input Format
- **Type:** Text (claim/news article)
- **Length:** 20-500 words recommended
- **Language:** English only
- **Format:** Plain text, UTF-8 encoding

### Output Format
- **Type:** Binary classification + confidence score
- **Labels:** 0 = Real/True, 1 = Fake/False
- **Confidence:** Probability score [0.0, 1.0]

### Example

**Input:**
```
"Scientists discover new renewable energy source that could power entire cities"
```

**Baseline Output:**
```json
{
  "prediction": 0,
  "confidence": 0.87,
  "reasoning": "TF-IDF score: 0.234"
}
```

**DistilBERT Output:**
```json
{
  "prediction": 1,
  "confidence": 0.62,
  "reasoning": "Low confidence - requires human review"
}
```

---

## Ethical Considerations

### Intended Use
- Assist fact-checkers and journalists
- Flag potentially false claims for human review
- Support content moderation workflows
- Educational research on misinformation

### Not Intended For
- Autonomous censorship without human oversight
- Determining individual credibility
- Replacing professional fact-checking
- Making binding decisions without review

### Limitations & Bias
- **Language:** English-only training data
- **Domain:** PolitiFact (politics-heavy), GossipCop (celebrity)
- **Temporal:** Training data from 2016-2017 (may be outdated)
- **False alarms:** Satire and sarcasm may be incorrectly flagged
- **Bias:** Models trained on fact-checker labels (inherit their biases)

### Recommended Deployment
- Always use with human review team
- Flag low-confidence predictions (0.45-0.55) for manual check
- Retrain periodically on new data
- Monitor for drift and performance degradation
- Disclose model limitations to users

---

## Known Limitations

1. **Domain Specificity**
   - Trained on PolitiFact/GossipCop data
   - May not generalize to other domains
   - Specialized topics (technical, scientific) may have degraded performance

2. **Temporal Bias**
   - Training data is 6+ years old
   - Language and misinformation tactics evolve
   - May be outdated for current events

3. **Language Coverage**
   - English only
   - Cannot detect misinformation in other languages
   - Code-switching texts may have poor performance

4. **Satire & Context**
   - Satire articles may be flagged as false
   - Out-of-context claims misclassified
   - Requires surrounding context for accuracy

5. **Text Length**
   - Optimized for 256 tokens (DistilBERT)
   - Very short claims (<10 words) less reliable
   - Very long articles (>500 words) truncated

---

## Performance Notes

### Generalization
- **Train-Val-Test Gap:** 2.5% (excellent generalization)
- **Cross-domain Performance:** Unknown (not tested)
- **Class Balance:** Works best with balanced datasets
- **Sample Size:** Best with 100+ predictions for statistical significance

### Reproducibility
- **Hardware:** Tesla T4 GPU, 16GB RAM
- **Framework Versions:** PyTorch 2.10, Transformers 4.x, Scikit-learn 1.x
- **Random Seed:** 42 (set in training script)
- **Results:** Reproducible within ±0.5% accuracy variance

---

## Usage Instructions

### Installation
```bash
# Clone repository
git clone https://github.com/sanjaykshetri/misinformation-at-scale.git
cd misinformation-at-scale

# Install dependencies
pip install -r requirements.txt
```

### Inference - Baseline Model
```python
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model
with open('models/baseline_lr_model.pkl', 'rb') as f:
    lr_model, vectorizer = pickle.load(f)

# Predict
text = "Your claim here"
features = vectorizer.transform([text])
prediction = lr_model.predict(features)[0]
confidence = lr_model.predict_proba(features)[0].max()

print(f"Prediction: {'Fake' if prediction else 'Real'} ({confidence:.1%})")
```

### Inference - DistilBERT Model
```python
import torch
from transformers import DistilBertTokenizer
from src.models import DistilBertClassifier

# Load model and tokenizer
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
    prediction = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1)[0][prediction].item()

print(f"Prediction: {'Fake' if prediction else 'Real'} ({confidence:.1%})")
```

---

## Model Card Acknowledgments

- **Dataset:** FakeNewsNet by Kai Shu & Suhang Wang
- **Base Model:** DistilBERT by Hugging Face
- **Training:** Google Colab (Tesla T4 GPU)
- **Evaluation:** Scikit-learn metrics

---

## References

### Academic Papers
1. Shu et al. (2020). "Fakenewsnet: A data repository for news content, social context and spatiotemporal information"
2. Sanh et al. (2019). "DistilBERT, a distilled version of BERT"
3. Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"

### Resources
- Project Repository: https://github.com/sanjaykshetri/misinformation-at-scale
- FakeNewsNet Data: https://github.com/KaiDMML/FakeNewsNet
- HuggingFace Models: https://huggingface.co/models

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-01 | Initial release with baseline + DistilBERT models |

---

**Last Updated:** April 1, 2026  
**Model Status:** ✅ Production Ready  
**Maintenance:** Active (feedback welcome via GitHub issues)
