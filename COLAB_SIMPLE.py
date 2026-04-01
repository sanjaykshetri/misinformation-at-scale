"""
GOOGLE COLAB - SIMPLE SETUP (Alternative approach)
Use if you get directory errors with other methods
"""

# ==============================================================================
# 🎯 COPY ALL CODE BELOW INTO ONE COLAB CELL AND RUN
# This approach doesn't use /tmp (simpler and more reliable)
# ==============================================================================

# Step 1: Install dependencies
!pip install -q pandas numpy scikit-learn torch transformers tqdm

# Step 2: Clone directly to current directory
!rm -rf misinformation-at-scale 2>/dev/null
!git clone https://github.com/sanjaykshetri/misinformation-at-scale.git
%cd misinformation-at-scale

# Step 3: Verify
import os
print("✓ Working directory:", os.getcwd())
print("✓ Scripts found:")
!ls -1 run_*.py

# Step 4: Check GPU
import torch
print(f"\n✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")

# Step 5: Run training
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80 + "\n")
exec(open('run_complete_training.py').read())
