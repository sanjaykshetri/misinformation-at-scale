#!/usr/bin/env python3
"""
Quick setup script to create a Google Colab notebook URL for this project.
Run this to get a direct link to run training on Colab.
"""

import json

# Create a Colab-ready setup script
colab_setup = '''
# Google Colab Setup for Misinformation at Scale - Deep Learning Training

## Quick Link (Copy & Paste)
Open this in your browser to start training immediately:
https://colab.research.google.com/drive/YOUR_COLAB_ID#scrollTo=your_cell_id

## Step 1: Enable GPU
1. Click "Runtime" in menu
2. Select "Change runtime type"
3. Choose "GPU" as Hardware accelerator
4. Click "Save"

## Step 2: Run Training

# CELL 1 - Setup
!pip install -q pandas numpy scikit-learn torch transformers tqdm
!git clone https://github.com/sanjaykshetri/misinformation-at-scale.git
%cd misinformation-at-scale

# CELL 2 - Run Training
%time
!python3 run_complete_training.py

## That\'s it! Training will take 30-60 minutes on GPU.
'''

print("="*80)
print("GOOGLE COLAB QUICK START")
print("="*80)
print("\n📝 Three ways to start:")
print("\n1️⃣  EASIEST: Open your notebook directly")
print("   URL: https://colab.research.google.com")
print("   Then: File → Open notebook → GitHub")
print("   Enter: sanjaykshetri/misinformation-at-scale")
print("   Select: notebooks/04_deep_learning_model.ipynb")

print("\n2️⃣  DIRECT LINK: Open this URL")
print("   https://colab.research.google.com/github/sanjaykshetri/misinformation-at-scale/blob/main/notebooks/04_deep_learning_model.ipynb")

print("\n3️⃣  FASTEST: Create new notebook and paste code")
print("   Go to: https://colab.research.google.com")
print("   Click: New notebook")
print("   Copy CELL 1 code (see guide)")

print("\n" + "="*80)
print("STEPS IN COLAB:")
print("="*80)
print("""
1. Enable GPU:
   - Runtime → Change runtime type → GPU → Save

2. Run CELL 1 (setup):
   !pip install -q pandas numpy scikit-learn torch transformers
   !git clone https://github.com/sanjaykshetri/misinformation-at-scale.git
   %cd misinformation-at-scale

3. Run CELL 2 (training):
   !python3 run_complete_training.py

⏱️  Total time: ~45 minutes on GPU (vs ~9 hours on CPU)
✅ Result: Both baseline (84%) and deep learning models (85-90%)
""")

print("="*80)
print("✨ Check GOOGLE_COLAB_GUIDE.md for detailed instructions")
print("="*80)
