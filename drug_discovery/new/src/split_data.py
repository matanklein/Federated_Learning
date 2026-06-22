"""
split_data.py
-------------
Splits data between clients with controlled task overlap.
Upgraded with Active-Row Stratification to prevent Dead Tasks.
Generates all 3 simulation environments: Full (P1+P2), P1 Self-Division, and P2 Self-Division.
"""
import os
import numpy as np
import packages.utils.data_utils as du

def stratified_idx_split(indices, ratio, Y_sparse):
    """Safely distributes active hits proportionally between two partitions."""
    num_total = len(indices)
    num_1 = int(ratio * (num_total / (ratio + 1)))
    
    # Identify which rows actually contain valuable hits within this specific subset
    Y_sub = Y_sparse[indices]
    Y_bin = (Y_sub > 0.5).astype(int)
    active_local_idxs = np.unique(Y_bin.nonzero()[0])
    inactive_local_idxs = np.setdiff1d(np.arange(num_total), active_local_idxs)
    
    np.random.shuffle(active_local_idxs)
    np.random.shuffle(inactive_local_idxs)
    
    # Give partition 1 its proportional share of the active rows
    u1_active_count = int(len(active_local_idxs) * (num_1 / num_total))
    
    part1_local = np.concatenate([
        active_local_idxs[:u1_active_count], 
        inactive_local_idxs[:num_1 - u1_active_count]
    ])
    part2_local = np.concatenate([
        active_local_idxs[u1_active_count:], 
        inactive_local_idxs[num_1 - u1_active_count:]
    ])
    
    np.random.shuffle(part1_local)
    np.random.shuffle(part2_local)
    
    return indices[part1_local], indices[part2_local]

def split_with_overlap(ratio, ecfp_tr, ic50_tr, root_dir="", overlap=2808):
    print(f"\n--- Splitting with {overlap} shared tasks (Stratified) ---")
    np.random.seed(42) # Ensure reproducible splits

    training_sample_num = ecfp_tr.shape[0]
    all_indices = np.arange(training_sample_num)
    
    # ==========================================
    # 1. Base Split: Divide Full Dataset into P1 and P2
    # ==========================================
    user_1, user_2 = stratified_idx_split(all_indices, ratio, ic50_tr)
    
    print(f"Total Base Split -> P1: {len(user_1)} samples | P2: {len(user_2)} samples")

    # ==========================================
    # 2. Handle Task Overlap and Disjunct Hiding (Masking)
    # ==========================================
    t_total = ic50_tr.shape[1]
    num_disjunct = (t_total - overlap) // 2
    print(f"Task Configuration -> {num_disjunct} disjunct, {overlap} shared (Total: {t_total})")

    shuffled_labels = np.random.permutation(t_total)
    common_labels = shuffled_labels[:overlap]

    disjunct_labels_1 = shuffled_labels[overlap:overlap + num_disjunct]
    disjunct_labels_2 = shuffled_labels[overlap + num_disjunct:overlap + 2 * num_disjunct]

    # Convert to CSC to efficiently wipe whole columns, eliminating false '0' predictions
    u1_ic50 = ic50_tr.tocsc()
    if len(disjunct_labels_2) > 0:
        u1_ic50[:, disjunct_labels_2] = 0
    u1_ic50.eliminate_zeros()
    u1_ic50 = u1_ic50.tocsr()

    u2_ic50 = ic50_tr.tocsc()
    if len(disjunct_labels_1) > 0:
        u2_ic50[:, disjunct_labels_1] = 0
    u2_ic50.eliminate_zeros()
    u2_ic50 = u2_ic50.tocsr()

    # ==========================================
    # 3. Generate FULL Environment (P1 vs P2)
    # ==========================================
    print("-> Generating FULL (P1 vs P2) environment splits...")
    full_dir = os.path.join(root_dir, "full/data_2_split/")
    os.makedirs(os.path.join(full_dir, "0_train/"), exist_ok=True)
    os.makedirs(os.path.join(full_dir, "0_test/"), exist_ok=True)
    os.makedirs(os.path.join(full_dir, "1_train/"), exist_ok=True)
    os.makedirs(os.path.join(full_dir, "1_test/"), exist_ok=True)

    # 80/20 Train/Test split is ratio = 4.0
    u1_tr, u1_te = stratified_idx_split(user_1, 4.0, u1_ic50)
    u2_tr, u2_te = stratified_idx_split(user_2, 4.0, u2_ic50)

    du.save_data(os.path.join(full_dir, "0_train/"), ecfp_tr[u1_tr], u1_ic50[u1_tr])
    du.save_data(os.path.join(full_dir, "0_test/"), ecfp_tr[u1_te], u1_ic50[u1_te])
    du.save_data(os.path.join(full_dir, "1_train/"), ecfp_tr[u2_tr], u2_ic50[u2_tr])
    du.save_data(os.path.join(full_dir, "1_test/"), ecfp_tr[u2_te], u2_ic50[u2_te])

    # ==========================================
    # 4. Generate P1 Self-Division (P1_1 vs P1_2)
    # ==========================================
    print("-> Generating P1 Self-Division environment splits...")
    p1_dir = os.path.join(root_dir, "p1/data_2_split/")
    os.makedirs(os.path.join(p1_dir, "0_train/"), exist_ok=True)
    os.makedirs(os.path.join(p1_dir, "0_test/"), exist_ok=True)
    os.makedirs(os.path.join(p1_dir, "1_train/"), exist_ok=True)
    os.makedirs(os.path.join(p1_dir, "1_test/"), exist_ok=True)

    # Split P1 in half (ratio = 1.0)
    p11, p12 = stratified_idx_split(user_1, 1.0, u1_ic50)
    
    # Split those halves into 80/20 Train/Test (ratio = 4.0)
    p11_tr, p11_te = stratified_idx_split(p11, 4.0, u1_ic50)
    p12_tr, p12_te = stratified_idx_split(p12, 4.0, u1_ic50)

    du.save_data(os.path.join(p1_dir, "0_train/"), ecfp_tr[p11_tr], u1_ic50[p11_tr])
    du.save_data(os.path.join(p1_dir, "0_test/"), ecfp_tr[p11_te], u1_ic50[p11_te])
    du.save_data(os.path.join(p1_dir, "1_train/"), ecfp_tr[p12_tr], u1_ic50[p12_tr])
    du.save_data(os.path.join(p1_dir, "1_test/"), ecfp_tr[p12_te], u1_ic50[p12_te])

    # ==========================================
    # 5. Generate P2 Self-Division (P2_1 vs P2_2)
    # ==========================================
    print("-> Generating P2 Self-Division environment splits...")
    p2_dir = os.path.join(root_dir, "p2/data_2_split/")
    os.makedirs(os.path.join(p2_dir, "0_train/"), exist_ok=True)
    os.makedirs(os.path.join(p2_dir, "0_test/"), exist_ok=True)
    os.makedirs(os.path.join(p2_dir, "1_train/"), exist_ok=True)
    os.makedirs(os.path.join(p2_dir, "1_test/"), exist_ok=True)

    # Split P2 in half (ratio = 1.0)
    p21, p22 = stratified_idx_split(user_2, 1.0, u2_ic50)
    
    # Split those halves into 80/20 Train/Test (ratio = 4.0)
    p21_tr, p21_te = stratified_idx_split(p21, 4.0, u2_ic50)
    p22_tr, p22_te = stratified_idx_split(p22, 4.0, u2_ic50)

    du.save_data(os.path.join(p2_dir, "0_train/"), ecfp_tr[p21_tr], u2_ic50[p21_tr])
    du.save_data(os.path.join(p2_dir, "0_test/"), ecfp_tr[p21_te], u2_ic50[p21_te])
    du.save_data(os.path.join(p2_dir, "1_train/"), ecfp_tr[p22_tr], u2_ic50[p22_tr])
    du.save_data(os.path.join(p2_dir, "1_test/"), ecfp_tr[p22_te], u2_ic50[p22_te])

    print("✅ Successfully generated structurally sound splits for Full, P1, and P2 environments.")