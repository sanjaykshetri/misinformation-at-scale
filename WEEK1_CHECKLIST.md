# Week 1 Milestone Checklist

## Project Foundation ✅
- [x] README.md (professional, research-grade tone)
- [x] requirements.txt (all dependencies pinned)
- [x] Project directory structure created
- [x] config/settings.yaml (comprehensive configuration)
- [x] DATA_GUIDE.md (complete data sourcing guide)
- [x] QUICK_START.md (quick reference guide)
- [x] prepare_data.py (data preparation utility)

## Core Ingestion Pipeline ✅
- [x] ingest.py (PySpark ingestion skeleton)
  - [x] RedditDataIngestion class with schema definition
  - [x] read_compressed_ndjson() method
  - [x] normalize_schema() method
  - [x] filter_by_subreddit() method
  - [x] filter_by_time_window() method
  - [x] remove_deleted_comments() method
  - [x] write_to_parquet() method
  - [x] Complete ingest_pipeline() orchestration
- [x] clean.py (data cleaning utilities)
  - [x] filter_by_length() for comment filtering
  - [x] filter_by_score() for quality control
  - [x] remove_urls() for text normalization
  - [x] create_class_label() for weak supervision
- [x] config.py (configuration management)

## Week 2 Preparation (Ready to Build)
- [ ] Notebook 01: 01_data_wrangling.ipynb
  - Load sample data from Parquet
  - EDA on raw comment distributions
  - Schema validation and statistics
  
- [ ] Notebook 02: 02_exploratory_analysis.ipynb
  - Class balance analysis
  - Text length distributions by class
  - Subreddit-level comparisons
  - Temporal trends
  
- [ ] Notebook 03: 03_baseline_modeling.ipynb
  - TF-IDF vectorization
  - Logistic regression baseline
  - Cross-validation and metrics
  - Feature importance analysis
  
- [ ] Notebook 04: 04_deep_learning_model.ipynb
  - Fine-tune DistilBERT
  - Training loop and evaluation
  - Error analysis
  - Model comparison vs baseline

## Testing & Validation
- [ ] Unit tests for ingest.py
- [ ] Unit tests for clean.py
- [ ] Integration test: end-to-end ingestion on sample data
- [ ] Validation: verify Parquet output schema

## Documentation
- [ ] Docstrings complete in all modules
- [ ] setup.py or pyproject.toml for package installation
- [ ] CONTRIBUTING.md (optional but recommended)
- [ ] .gitignore (data/, logs/, outputs/)

## Infrastructure
- [ ] .env template for local configuration
- [ ] logs/ directory created
- [ ] reports/ directory for outputs
- [ ] Download guide for Reddit data in README.md

---

## Week 1 Priorities (This Week)

**Phase 1: Foundation (COMPLETE)** ✅
1. README, requirements.txt, directory structure
2. ingest.py skeleton with full documentation
3. clean.py utilities
4. config.py and settings.yaml

**Phase 2: Validation (NEXT)**
1. Set up a small sample Reddit dataset (from academic torrent or synthetic)
2. Test ingest.py end-to-end on sample data
3. Verify Parquet output and schema correctness
4. Document any issues found

**Phase 3: Notebooks (START)**
1. Create 01_data_wrangling.ipynb with sample data
2. Write basic EDA cells
3. Load from Parquet and show summary statistics

---

## Notes
- All code is documented and ready for Springboard code review
- Architecture supports scaling to multi-gigabyte datasets
- Weak labeling strategy is explicit and defensible
- Config-driven design allows easy experiment iteration
