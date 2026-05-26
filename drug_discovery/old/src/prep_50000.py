# this code aims to get a small partition of the targeted dataset, for quick run purposes
import sys
from pathlib import Path

# Ensure project root (parent of `src`) is on sys.path so `packages` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import packages.utils.data_utils as du
import scipy.sparse
import numpy as np
import os

# Use the actual full dataset location and write outputs to `data_quarter/` in project root
project_root = Path(__file__).resolve().parents[1]
data_path = str(project_root / "data") + os.path.sep
out_root = str(project_root / "data_quarter") + os.path.sep
SEED       = 42

ecfp_tr, ic50_tr, ecfp_va, ic50_va = du.load_data(data_path)
ecfp_tr = du.fold_input(ecfp_tr, 32000)
ecfp_va = du.fold_input(ecfp_va, 32000)

N = int(ecfp_tr.shape[0] * 0.25) # 25% of the training samples

rng  = np.random.RandomState(SEED)
idx  = rng.choice(ecfp_tr.shape[0], size=N, replace=False)
idx.sort()

ecfp_50k = ecfp_tr[idx]
ic50_50k = ic50_tr[idx]

os.makedirs(out_root, exist_ok=True)
scipy.sparse.save_npz(out_root + "x_tr.npz", ecfp_50k)
scipy.sparse.save_npz(out_root + "y_tr.npz", ic50_50k)
scipy.sparse.save_npz(out_root + "x_va.npz", ecfp_va)
scipy.sparse.save_npz(out_root + "y_va.npz", ic50_va)
print(f"Saved {N} samples → {out_root}")