import random
import packages.utils.data_utils as du
import sparsechem as sc

def hide_h_percent(X_train, Y_train, h):
    if h == 0.0:
        return X_train, Y_train
    X_train_num = X_train.shape[0]
    remaining_data_num = int((1 - h) * X_train_num)
    shared_data_indices = random.sample(range(X_train_num), remaining_data_num)
    return X_train[shared_data_indices], Y_train[shared_data_indices]

def get_client_datasets(data_path, k, privacy_param, privacy_mode, conf):
    """
    privacy_mode: 'suppression' or 'dp'
    privacy_param: h (hide percentage) or p (noise scale)
    """
    X_train, Y_train = du.load_ratio_split_data(data_path, k, train=True)
    
    # Apply suppression only if mode matches
    if privacy_mode == 'suppression' and privacy_param > 0.0:
        X_train, Y_train = hide_h_percent(X_train, Y_train, privacy_param)
        
    train_dataset = sc.SparseDataset(X_train, Y_train)
    conf.output_size = Y_train.shape[1]

    X_test, Y_test = du.load_ratio_split_data(data_path, k, train=False)
    test_dataset = sc.SparseDataset(X_test, Y_test)

    return train_dataset, test_dataset