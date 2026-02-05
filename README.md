# Misinformation at Scale
### Large-scale NLP analysis of misinformation patterns in Reddit discourse using PySpark and deep learning

---

## Overview

This project investigates how misinformation-like discourse emerges and propagates in large online communities. Using multi-gigabyte Reddit comment datasets, the project builds a distributed data pipeline with PySpark to ingest, clean, and analyze large-scale conversational text. It then applies modern NLP and deep learning techniques to model linguistic patterns associated with misinformation-related communities.

The goal is not to declare absolute truth or falsehood, but to detect **statistical signals correlated with misinformation discourse** using transparent labeling strategies and scalable machine learning methods.

This repository serves as a portfolio-grade demonstration of:

- Distributed big-data engineering
- Large-scale NLP pipelines
- Weakly supervised labeling strategies
- Deep learning text classification
- Research-driven exploratory data analysis

---

## Research Question

Can large-scale linguistic and conversational features predict misinformation-associated discourse patterns in online communities?

---

## Dataset

This project uses publicly available Reddit comment archives distributed via academic torrents. The dataset consists of compressed NDJSON files containing Reddit submissions and comments across multiple subreddits and time windows.

To keep the project computationally tractable while maintaining big-data scale, the analysis focuses on a selected subset of subreddits and a defined time window.

### Data characteristics

- Multi-gigabyte text corpus
- Nested JSON comment structure
- Rich metadata (timestamps, subreddit, author, score, etc.)
- Community-level structure suitable for weak labeling

**Note:** Raw data files are not stored in this repository due to size. See the setup section for download instructions.

---

## Labeling Strategy

Because misinformation does not come with universal ground-truth labels, this project uses a **community-based weak supervision approach**.

Comments are labeled according to subreddit affiliation:

- Communities associated with misinformation or conspiracy discourse
- Control communities focused on science or neutral discussion

This approach models **signals correlated with misinformation**, not definitive truth judgments. All assumptions are documented and evaluated during exploratory analysis.

---

## Project Architecture

### 1. Data Ingestion (PySpark)

- Stream reading compressed Reddit dumps
- Schema normalization across files
- Filtering by subreddit and time window
- Writing cleaned datasets to Parquet format

### 2. Data Wrangling

- Handling missing and deleted content
- Removing duplicates and corrupted entries
- Text normalization and preprocessing
- Feature engineering for modeling

### 3. Exploratory Data Analysis

- Class balance inspection
- Text length and distribution analysis
- Temporal activity trends
- Community-level comparisons
- Linguistic feature exploration

### 4. Modeling

#### Baseline models

- TF-IDF vectorization
- Logistic regression / linear classifiers (Spark ML)

#### Deep learning models

- Transformer-based text classification
- Fine-tuned neural architectures
- Comparative evaluation

### 5. Evaluation

- Precision / recall / F1 metrics
- Confusion matrices
- Error analysis
- Model interpretability discussion

---

## Repository Structure

```
misinformation-at-scale/
│
├── notebooks/
│   ├── 01_data_wrangling.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_baseline_modeling.ipynb
│   └── 04_deep_learning_model.ipynb
│
├── src/
│   ├── ingest.py
│   ├── clean.py
│   ├── features.py
│   └── modeling.py
│
├── config/
│   └── settings.yaml
│
├── requirements.txt
└── README.md
```

---

## Setup

### Environment

Recommended:

- Python 3.10+
- Apache Spark 3.x
- PySpark
- PyTorch
- Hugging Face Transformers

Install dependencies:

```bash
pip install -r requirements.txt
```

### Data Setup

**See [DATA_GUIDE.md](DATA_GUIDE.md) for comprehensive instructions on:**
- Where to download Reddit data (Academic Torrents)
- Data format and directory structure
- Decompression and verification
- Quick start with helper scripts

**Quick steps:**

1. Download Reddit comments from [Academic Torrents](https://academictorrents.com/) (search "RC_2020")
2. Place `.bz2` or `.gz` files in `data/raw/`
3. Run data preparation:

```bash
# Create directories and decompress files
python prepare_data.py --all
```

4. Run ingestion pipeline:

```bash
python src/ingest.py
```

Cleaned datasets will be written to `data/processed/`

**No data yet?** Notebooks automatically generate synthetic data for testing the pipeline.

---

## Usage

Run notebooks in order:

1. Data wrangling
2. Exploratory data analysis
3. Baseline modeling
4. Deep learning modeling

Each notebook documents assumptions, decisions, and findings.

---

## Ethical Considerations

This project analyzes publicly available online discourse. It does not attempt to assign moral judgments to individuals or communities. The goal is to study statistical patterns in language and communication at scale.

All labeling strategies are explicitly documented and interpreted cautiously.

---

## Skills Demonstrated

* Distributed data processing with PySpark
* Large-scale text ingestion and cleaning
* NLP feature engineering
* Weakly supervised learning
* Deep learning model training
* Research-oriented exploratory analysis
* Reproducible pipeline design

---

## Future Work

* Graph-based misinformation propagation modeling
* Conversation-aware neural architectures
* Cross-platform misinformation comparison
* Explainable AI for linguistic interpretation

---

## Author

Sanjay Chhetri

---

## License

MIT License
