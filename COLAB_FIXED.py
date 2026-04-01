"""
GOOGLE COLAB - WORKING SETUP FOR MISINFORMATION AT SCALE
Fixed version that handles existing directories

⚠️ IMPORTANT: If you get "directory exists" error, use THIS code instead
"""

# ==============================================================================
# 🎯 COPY ALL CODE BELOW INTO ONE COLAB CELL AND RUN
# ==============================================================================

import os
import shutil

# Step 0: Clean up old directories (IMPORTANT to avoid "directory exists" error)
print("Cleaning up old directories from previous runs...")
for old_dir in ['/tmp/training', '/tmp/ml_training', '/tmp/fakenews']:
    if os.path.exists(old_dir):
        print(f"  ✓ Removing {old_dir}")
        shutil.rmtree(old_dir, ignore_errors=True)

# Step 1: Install dependencies
print("\nInstalling dependencies...")
!pip install -q pandas numpy scikit-learn torch transformers tqdm

# Step 2: Clone repository to fresh location
print("Cloning repository...")
!git clone https://github.com/sanjaykshetri/misinformation-at-scale.git /tmp/training
%cd /tmp/training

# Step 3: Verify clone worked
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)
print("✓ Current directory:", os.getcwd())
print("✓ Files present:")
!ls -1 run_*.py 2>/dev/null || echo "ERROR: Scripts not found!"

# Step 4: Check GPU
import torch
print("\n✓ PyTorch:", torch.__version__)
gpu_available = torch.cuda.is_available()
print("✓ GPU available:", "YES ✅" if gpu_available else "NO ❌ (training will be slow)")
if gpu_available:
    print(f"✓ GPU type: {torch.cuda.get_device_name(0)}")

# Step 5: Run training
print("\n" + "="*80)
if gpu_available:
    print("🚀 STARTING DEEP LEARNING TRAINING ON GPU")
    print("   Expected time: 30-60 minutes")
else:
    print("⚠️  TRAINING ON CPU (slow - consider using GPU)")
print("="*80 + "\n")

# Run the complete training script
exec(open('run_complete_training.py').read())
