import os

import numpy as np

import packages.utils.data_utils as du


def split_with_overlap(ratio, ecfp_tr, ic50_tr, root_dir="", overlap=1000):
    """
    Split data into two clients with balanced task coverage and label density.
    Ensures equal number of active tasks per client and saves train/test splits.
    """
    print(f"Split with {overlap} overlap (balanced).")

    split_path = os.path.join(root_dir, "data_2_split/")
    os.makedirs(split_path, exist_ok=True)

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

    t_total = ic50_tr.shape[1]
    num_disjunct = (t_total - overlap) // 2
    print(f"{num_disjunct} disjunct and {overlap} overlapping labels (total: {t_total})")

    shuffled_labels = np.random.permutation(t_total)
    common_labels = shuffled_labels[:overlap]

    disjunct_labels_1 = shuffled_labels[overlap:overlap + num_disjunct]
    disjunct_labels_2 = shuffled_labels[overlap + num_disjunct:overlap + 2 * num_disjunct]

    user_1_labels = np.concatenate([common_labels, disjunct_labels_1])
    user_2_labels = np.concatenate([common_labels, disjunct_labels_2])

    user_1_labels = np.sort(user_1_labels)
    user_2_labels = np.sort(user_2_labels)

    u1_ic50 = ic50_tr.tocsc()[:, user_1_labels].tocsr()
    u2_ic50 = ic50_tr.tocsc()[:, user_2_labels].tocsr()

    du.save_data(os.path.join(split_path, "0_train/"), ecfp_tr[user_1_train], u1_ic50[user_1_train])
    du.save_data(os.path.join(split_path, "0_test/"), ecfp_tr[user_1_test], u1_ic50[user_1_test])

    du.save_data(os.path.join(split_path, "1_train/"), ecfp_tr[user_2_train], u2_ic50[user_2_train])
    du.save_data(os.path.join(split_path, "1_test/"), ecfp_tr[user_2_test], u2_ic50[user_2_test])
