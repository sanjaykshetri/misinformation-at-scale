# 📊 Discourse Classification Dashboard

**Interactive web interface for testing the misinformation/discourse classifier model**

## Overview

This dashboard provides an interactive interface to:
- ✅ **Classify text** and see predictions with confidence scores
- 🧪 **Test adversarial examples** to probe model robustness
- 📊 **Analyze feature importance** and model behavior  
- 📈 **Review performance metrics** across all datasets

## Features

### 1. 🎯 Interactive Classifier
- Enter custom text for real-time classification
- View confidence scores and uncertainty levels
- See top predictive features for each prediction
- Display text statistics (word count, sentence count, etc.)

### 2. 🧪 Adversarial Testing
Test model robustness with adversarial examples:
- **Keyword swap** (vaccine → chemotherapy)
- **Uncertainty injection** (adding hedge words)
- **Authority flip** (credibility manipulation)
- **Negation** (reversing claims)
- **Emotional language** (adding charged terms)
- **Spelling noise** (typos and text corruption)

Visualize how predictions change under perturbations.

### 3. 📊 Model Analysis
- Top 20 features predictive of each class
- Vocabulary separation analysis
- Feature overlap statistics
- Implications for real-world performance

### 4. 📈 Performance Metrics
- Training vs. test set comparison
- 5-fold cross-validation results
- Confusion matrix visualization
- Performance charts and graphs

---

## Installation

### 1. Install Streamlit
```bash
pip install -r dashboard_requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

### 2. Navigate to Project
```bash
cd c:\Users\sanja\OneDrive\Documents\GitHub\misinformation-at-scale
```

---

## Running the Dashboard

### Quick Start
```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

### Custom Port
```bash
streamlit run app.py --server.port 8502
```

### With Logging
```bash
streamlit run app.py --logger.level=debug
```

---

## Dashboard Pages

### 🎯 Interactive Classifier
**Best for**: Testing the model on custom text

**Actions**:
1. Enter text in the text area
2. Adjust confidence threshold (optional)
3. Click "Classify" button
4. View results with feature importance

**Example inputs**:
```
"Vaccines have been proven effective in clinical trials"
"COVID vaccines cause blood clots and heart problems"
"Multiple studies show vaccines reduce hospitalization by 85%"
```

### 🧪 Adversarial Testing
**Best for**: Understanding model robustness

**What it shows**:
- Original text prediction
- 6 different adversarial variations
- How predictions change with each perturbation
- Confidence score changes

**Key insight**: 
Model predictions change significantly when keywords are swapped or negated, revealing heavy reliance on specific vocabulary rather than semantic understanding.

### 📊 Model Analysis
**Best for**: Understanding what the model learned

**Includes**:
- Feature importance bar charts (side by side comparison)
- Vocabulary separation metrics
- Overlap analysis (% of shared terms)
- Implications for real-world deployment

### 📈 Performance Metrics
**Best for**: Reviewing model quality

**Shows**:
- All standard metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Cross-validation fold results
- Confusion matrix heatmap
- Performance comparison charts

---

## Understanding the Results

### ✅ Perfect Accuracy (100%)
The model achieves 100% accuracy because:
- **Vocabulary is completely separated** between classes
- Factual claims use: "studies", "evidence", "experts", "data"
- Misinformation uses: "wake", "censored", "gates", "lying"
- Model learns vocabulary → classification, not claim verification

### ⚠️ Adversarial Vulnerability
The model is vulnerable to:
- **Keyword swaps**: Changing domain-specific terms
- **Negation**: Flipping claim polarity
- **Emotional language**: Adding charged terms
- **Spelling noise**: Typos and misspellings

This indicates the model **cannot** handle real-world misinformation where:
- Both true and false claims discuss the same topic
- Same vocabulary is used for competing claims
- Subtle false information requires evidence checking

---

## Model Specifications

| Aspect | Value |
|--------|-------|
| **Base Model** | Logistic Regression |
| **Vectorizer** | TF-IDF |
| **Vocabulary Size** | 369 features |
| **Training Data** | Discourse Classification (7,000 examples) |
| **Train/Val/Test** | 70% / 15% / 15% |
| **Classes** | Factual (0) vs. Misinformation (1) |
| **Test Accuracy** | 100% |

---

## Interpreting Feature Importance

### For Factual Claims
**Top Features**: real, safe, recommends, studies, experts, data, evidence

These words indicate:
- Scientific/academic framing
- Evidence-based language
- Authority references
- Data-driven claims

### For Misinformation
**Top Features**: wake, censored, lying, truth, don, shock, gates

These words indicate:
- Conspiracy framing
- Secretive/censorship language
- Emotional appeals
- Specific cultural references

---

## Caveats & Limitations

⚠️ **Important Notes**:

1. **Perfect accuracy is unrealistic**
   - Real misinformation has vocabulary overlap with factual claims
   - Model performance would be much lower (~70-80%) with realistic data

2. **Vocabulary-separated data**
   - This dataset was synthetically generated with clear separation
   - Real-world claims don't have such clean boundaries

3. **No semantic understanding**
   - Model cannot verify claims against facts
   - Relies purely on linguistic patterns
   - Would fail on novel misinformation styles

4. **Limited to training domain**
   - Only trained on vaccines, elections, climate
   - May not generalize to other topics

---

## Future Improvements

- [ ] Load actual trained models from checkpoint files
- [ ] Add LIME/SHAP explanations for deeper interpretability
- [ ] Implement adversarial training analysis
- [ ] Add model comparison (TF-IDF vs. DistilBERT)
- [ ] Create attention visualizations for transformer models
- [ ] Add confidence calibration metrics
- [ ] Implement batch prediction support
- [ ] Add model update/retraining interface

---

## Troubleshooting

### Dashboard won't start
```bash
# Clear Streamlit cache
streamlit cache clear

# Try with verbose output
streamlit run app.py --logger.level=debug
```

### Port already in use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Slow performance
- Reduce text length
- Clear cache between predictions
- Check available RAM
- Monitor CPU usage

---

## Contact & Documentation

- 📖 **Full Analysis**: See `DATASET_ANALYSIS.md`
- 📊 **Model Details**: See `PRACTICAL_OPTIONS.md`
- 🔬 **Notebooks**: See `notebooks/` folder
  - `01_data_wrangling.ipynb` - Data loading & preprocessing
  - `02_exploratory_analysis.ipynb` - Linguistic analysis
  - `03_baseline_modeling.ipynb` - TF-IDF + LogReg training
  - `04_deep_learning_modeling.ipynb` - DistilBERT fine-tuning

---

**Created**: February 2026  
**Status**: Active Development  
**License**: Educational Use
