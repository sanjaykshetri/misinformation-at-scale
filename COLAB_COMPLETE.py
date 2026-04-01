"""
GOOGLE COLAB - COMPLETE FIX (Data + Training)
This version downloads/creates the data files before training
"""

# ==============================================================================
# 🎯 COPY ALL CODE BELOW INTO ONE COLAB CELL AND RUN
# ==============================================================================

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
# This creates the required CSV files
exec(open('load_fakenewsnet.py').read())

print("\nStep 5/5: Verifying data files...")
import os
data_files = [
    'data/processed/fakenewsnet_train.csv',
    'data/processed/fakenewsnet_val.csv',
    'data/processed/fakenewsnet_test.csv'
]
for f in data_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024*1024)  # Size in MB
        print(f"✓ {f} ({size:.1f} MB)")
    else:
        print(f"❌ {f} NOT FOUND")

# Check GPU
import torch
print(f"\n✓ GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

print("\n" + "="*80)
print("🚀 STARTING COMPLETE TRAINING")
print("="*80 + "\n")

# Run complete training
exec(open('run_complete_training.py').read())

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
