import random
import numpy as np
import packages.utils.data_utils as du
import os
'''
data splitting with controlled task overlap:

- splits data between 2 clients with a specified ratio
- controls how many tasks are shared (overlap parameter)
- balanced task distribution
- creates train/test splits (80/20) for each client
- handles both sample distribution and task/label distribution
'''


def split_with_overlap(ratio, ecfp_tr, ic50_tr, root_dir="", overlap=2808):
    print(f"Split with {overlap} overlap (balanced).")
    
    # Sample distribution
    training_sample_num = ecfp_tr.shape[0]
    one_part = int(training_sample_num / (ratio + 1))
    samples_user_1 = ratio * one_part
    samples_user_2 = one_part
    print(f"Number of training samples user-1: {samples_user_1}")
    print(f"Number of training samples user-2: {samples_user_2}")

    shuffled_idx = np.random.permutation(training_sample_num)
    user_1 = shuffled_idx[:samples_user_1]
    user_2 = shuffled_idx[-samples_user_2:]

    user_1_train_size = int(0.8 * samples_user_1)
    user_2_train_size = int(0.8 * samples_user_2)

    user_1_train = user_1[:user_1_train_size]
    user_1_test = user_1[user_1_train_size:]
    user_2_train = user_2[:user_2_train_size]
    user_2_test = user_2[user_2_train_size:]

    # Label/task assignment
    T_total = ic50_tr.shape[1]
    num_disjunct = (T_total - overlap) // 2
    print(f"{num_disjunct} disjunct and {overlap} overlapping labels (total: {T_total})")
    
    shuffled_labels = np.random.permutation(T_total)
    common_labels = shuffled_labels[:overlap]

    disjunct_labels_1 = shuffled_labels[overlap:overlap + num_disjunct]
    disjunct_labels_2 = shuffled_labels[overlap + num_disjunct:overlap + 2 * num_disjunct]

    user_1_labels = np.concatenate([common_labels, disjunct_labels_1])
    user_2_labels = np.concatenate([common_labels, disjunct_labels_2])

    user_1_labels = np.sort(user_1_labels)
    user_2_labels = np.sort(user_2_labels)

    u1_ic50 = ic50_tr.tocsc()[:, user_1_labels].tocsr()
    u2_ic50 = ic50_tr.tocsc()[:, user_2_labels].tocsr()

    # ---------------------------------------------------------
    # 1. Save FULL setup (Player 1 + Player 2 Federated)
    # ---------------------------------------------------------
    full_dir = os.path.join(root_dir, "full/data_2_split/")
    os.makedirs(full_dir, exist_ok=True)
    du.save_data(os.path.join(full_dir, "0_train/"), ecfp_tr[user_1_train], u1_ic50[user_1_train])
    du.save_data(os.path.join(full_dir, "0_test/"), ecfp_tr[user_1_test], u1_ic50[user_1_test])
    du.save_data(os.path.join(full_dir, "1_train/"), ecfp_tr[user_2_train], u2_ic50[user_2_train])
    du.save_data(os.path.join(full_dir, "1_test/"), ecfp_tr[user_2_test], u2_ic50[user_2_test])

    # ---------------------------------------------------------
    # 2. Save P1 setup (Solo P1 split into 2 pseudo-clients)
    # ---------------------------------------------------------
    p1_dir = os.path.join(root_dir, "p1/data_2_split/")
    os.makedirs(p1_dir, exist_ok=True)
    
    # Split P1 into two equal halves
    half_p1 = len(user_1) // 2
    p1_c0 = user_1[:half_p1]
    p1_c1 = user_1[half_p1:]
    
    p1_c0_tr, p1_c0_te = p1_c0[:int(0.8 * len(p1_c0))], p1_c0[int(0.8 * len(p1_c0)):]
    p1_c1_tr, p1_c1_te = p1_c1[:int(0.8 * len(p1_c1))], p1_c1[int(0.8 * len(p1_c1)):]

    du.save_data(os.path.join(p1_dir, "0_train/"), ecfp_tr[p1_c0_tr], u1_ic50[p1_c0_tr])
    du.save_data(os.path.join(p1_dir, "0_test/"), ecfp_tr[p1_c0_te], u1_ic50[p1_c0_te])
    du.save_data(os.path.join(p1_dir, "1_train/"), ecfp_tr[p1_c1_tr], u1_ic50[p1_c1_tr])
    du.save_data(os.path.join(p1_dir, "1_test/"), ecfp_tr[p1_c1_te], u1_ic50[p1_c1_te])

    # ---------------------------------------------------------
    # 3. Save P2 setup (Solo P2 split into 2 pseudo-clients)
    # ---------------------------------------------------------
    p2_dir = os.path.join(root_dir, "p2/data_2_split/")
    os.makedirs(p2_dir, exist_ok=True)
    
    # Split P2 into two equal halves
    half_p2 = len(user_2) // 2
    p2_c0 = user_2[:half_p2]
    p2_c1 = user_2[half_p2:]
    
    p2_c0_tr, p2_c0_te = p2_c0[:int(0.8 * len(p2_c0))], p2_c0[int(0.8 * len(p2_c0)):]
    p2_c1_tr, p2_c1_te = p2_c1[:int(0.8 * len(p2_c1))], p2_c1[int(0.8 * len(p2_c1)):]

    du.save_data(os.path.join(p2_dir, "0_train/"), ecfp_tr[p2_c0_tr], u2_ic50[p2_c0_tr])
    du.save_data(os.path.join(p2_dir, "0_test/"), ecfp_tr[p2_c0_te], u2_ic50[p2_c0_te])
    du.save_data(os.path.join(p2_dir, "1_train/"), ecfp_tr[p2_c1_tr], u2_ic50[p2_c1_tr])
    du.save_data(os.path.join(p2_dir, "1_test/"), ecfp_tr[p2_c1_te], u2_ic50[p2_c1_te])
    
    print(f"Successfully generated 18 directories and 24 files inside {root_dir}")