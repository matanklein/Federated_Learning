import random
import packages.utils.data_utils as du
import sparsechem as sc
import numpy as np

def hide_h_percent(X_train, Y_train, h, seed=42):
    if h == 0.0:
        return X_train, Y_train
    rng = np.random.RandomState(seed)
    X_train_num = X_train.shape[0]
    remaining_data_num = int((1.0 - h) * X_train_num)
    
    # We use simple random hiding here, because the overall data
    # was already rigorously stratified in prep_subsets.py
    shared_data_indices = rng.choice(X_train_num, size=remaining_data_num, replace=False)
    shared_data_indices.sort()
    
    return X_train[shared_data_indices], Y_train[shared_data_indices]

def get_client_datasets(data_path, k, privacy_param, privacy_mode, conf, seed=42):
    X_train, Y_train = du.load_ratio_split_data(data_path, k, train=True)
    
    if privacy_mode == 'sup' and privacy_param > 0.0:
        X_train, Y_train = hide_h_percent(X_train, Y_train, privacy_param, seed=seed)
        
    train_dataset = sc.SparseDataset(X_train, Y_train)
    conf.output_size = Y_train.shape[1]

    X_test, Y_test = du.load_ratio_split_data(data_path, k, train=False)
    test_dataset = sc.SparseDataset(X_test, Y_test)

    return train_dataset, test_dataset