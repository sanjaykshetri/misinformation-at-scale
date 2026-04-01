"""
GOOGLE COLAB - WORKING SETUP FOR MISINFORMATION AT SCALE

This file contains the exact code to copy-paste into Google Colab.
It fixes the "file not found" error by using proper paths.
"""

# ==============================================================================
# 🎯 COPY ALL CODE BELOW INTO ONE COLAB CELL
# ==============================================================================
# Then click the ▶️ button to run

# Step 1: Install dependencies
!pip install -q pandas numpy scikit-learn torch transformers tqdm

# Step 2: Clone repository to a known location
!git clone https://github.com/sanjaykshetri/misinformation-at-scale.git /tmp/ml_training
%cd /tmp/ml_training

# Step 3: Verify everything is there
import os
print("✓ Current directory:", os.getcwd())
print("✓ Files present:")
!ls -1 run_*.py

# Step 4: Check GPU
import torch
print("\n✓ PyTorch:", torch.__version__)
print("✓ GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"✓ GPU type: {torch.cuda.get_device_name(0)}")

# Step 5: Run the training script
print("\n" + "="*80)
print("STARTING COMPLETE TRAINING - This takes 30-60 minutes")
print("="*80 + "\n")

exec(open('run_complete_training.py').read())

# ==============================================================================
# END - All output will appear below
# ==============================================================================
