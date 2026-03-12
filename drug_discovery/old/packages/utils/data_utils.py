import scipy.io
import numpy as np
import sparsechem as sc
import os
import random
from functools import reduce
import scipy.sparse as sparse

def load_data(rel_path, folding_split=True, train_ratio=0.8):
    """
    Parameters
    ----------
    rel_path : str
        Relative path to data.
    folding_split : bool
        Whether to split data based on folding.
    train_ratio : float
        Ratio of training data. Default value is 0.8, which results in an 80%-20% split.

    Returns
    -------
    scipy.sparse.csr_matrix
        A sparse matrix containing the training data.
    scipy.sparse.csr_matrix
        A sparse matrix containing the training data labels.
    scipy.sparse.csr_matrix
        A sparse matrix containing the validation data.
    scipy.sparse.csr_matrix
        A sparse matrix containing the validation data labels.
    """
    ecfp    = scipy.io.mmread(rel_path + "chembl_23_x.mtx").tocsr()
    ic50    = scipy.io.mmread(rel_path + "chembl_23_y.mtx").tocsr()

    if folding_split:
        folding = np.load(rel_path + "folding_hier_0.6.npy")
        fold_va = 0
        idx_tr  = np.where(folding != fold_va)[0]
        idx_va  = np.where(folding == fold_va)[0]

        ecfp_tr = ecfp[idx_tr]
        ic50_tr = ic50[idx_tr]

        ecfp_va = ecfp[idx_va]
        ic50_va = ic50[idx_va]

        del folding
    else:
        num_samples = ecfp.shape[0]
        num_tr_samples = int(num_samples * train_ratio)
        num_te_samples = int(num_samples * (1-train_ratio))
        shuffled_idx = np.array(range(num_samples))
        np.random.shuffle(shuffled_idx)
        idx_tr = shuffled_idx[:num_tr_samples]
        idx_va = shuffled_idx[-num_te_samples:]

        ecfp_tr = ecfp[idx_tr]
        ic50_tr = ic50[idx_tr]

        ecfp_va = ecfp[idx_va]
        ic50_va = ic50[idx_va]

    del ecfp
    del ic50

    return ecfp_tr, ic50_tr, ecfp_va, ic50_va

def load_split_data(rel_path, k):
    """
    Loads the k'th client's data from rel_path.

    Parameters
    ----------
    rel_path : str
        Relative path where to read the client data.
    k : int
        The index of the client.

    Returns
    -------
    scipy.sparse.csr_matrix
        A sparse matrix containing the data.
    scipy.sparse.csr_matrix
        A sparse matrix containing the data labels.
    """
    # X_data = scipy.io.mmread(rel_path + str(k) + "/X_data.mtx").tocsr()
    # Y_data = scipy.io.mmread(rel_path + str(k) + "/Y_data.mtx").tocsr()
    X_data = scipy.io.mmread(os.path.join(rel_path, f"{k}_train", "X_data.mtx")).tocsr()
    Y_data = scipy.io.mmread(os.path.join(rel_path, f"{k}_train", "Y_data.mtx")).tocsr()


    return X_data, Y_data


def load_ratio_split_data(rel_path, k, train):
    if train:
        path = rel_path + str(k) + "_train"
    else:
        path = rel_path + str(k) + "_test"
    X_data = scipy.io.mmread(path + "/X_data.mtx").tocsr()
    Y_data = scipy.io.mmread(path + "/Y_data.mtx").tocsr()

    return X_data, Y_data

def chunk_list(list_, N_):
    """
    Chunks list_ into N_ equal sized chunks.

    Parameters
    ----------
    list_ : list
        A list of elements.
    N_ : int
        Number of chunks.

    Returns
    -------
    list
        List of created chunks.
    """
    res = []
    chunk_length = int(len(list_)/N_)
    k=0

    for i in range(N_):
        res.append(list_[k:k+chunk_length])
        k+=chunk_length

    return res

def save_data(rel_path, data, labels):
    """
    Saves data and corresponding labels to rel_path. Creates rel_path if it
    doesn't exist.

    Parameters
    ----------
    rel_path : str
        Relative path to directory.
    data :  scipy.sparse.csr_matrix
        Data to save.
    labels : scipy.sparse.csr_matrix
        Labels to save.
    """
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)

    scipy.io.mmwrite(rel_path + "X_data.mtx", data)
    scipy.io.mmwrite(rel_path + "Y_data.mtx", labels)

def make_split_save(k, ecfp_tr, ic50_tr, root_dir="", target_split=False):
    """
    Make a random k-fold split in the training samples and save them under
    data_<k>_split/.

    Parameters
    ----------
    k : int
        Number of splits.
    ecfp_tr : scipy.sparse.csr_matrix
        Training data.
    ic50_tr : scipy.sparse.csr_matrix
        Labels.
    root_dir : str
        Path where to create the folder with the splits.
    target_split : bool
        Whether the targets should be split amongst the clients. Default value
        is "False": in this case all clients know the bioactivity of compounds
        with all of the targets.
    """
    print("Making a random %d-fold split in the samples"%(k))
    
    split_path = root_dir + "data_" + str(k) + "_split/"
    
    training_sample_num = ecfp_tr.shape[0]
    samples_per_user = int(training_sample_num/k)
    print("Number of samples per user: %d"%(samples_per_user))
    
    ## sample samples_per_user*k indices
    samples = random.sample(range(training_sample_num), samples_per_user*k)
    ## divide sample list into k equal chunks 
    batch_indices = chunk_list(samples, k)

    if target_split:
        # T_total - number of targets
        T_total = ic50_tr.shape[1]
        # t_c - targets per client
        t_c = int(T_total/k)
        targets_split = [(from_, to_) for from_, to_ in zip(range(0,T_total, t_c),
            range(t_c, T_total+t_c, t_c))]

        print("Targets split length: ", len(targets_split))
        print("Targets of client number <i>: (<from>, <to>)")
        for i in range(k):
                print("Targets of client number %d: %s" % (i, targets_split[i]))

        ## split and save training data
        print("Saving split data in '%s'" % split_path)
        for i_indice, from_to in zip(enumerate(batch_indices), targets_split):
            save_data(split_path + str(i_indice[0]) + "/", ecfp_tr[i_indice[1]],
                    ic50_tr[i_indice[1], from_to[0]:from_to[1]])
    else:
        ## split and save training data
        print("Saving split data in '%s'" % split_path)
        for i, indice in enumerate(batch_indices):
            save_data(split_path + str(i) + "/", ecfp_tr[indice], ic50_tr[indice])


def make_ratio_split_save(ratio, ecfp_tr, ic50_tr, root_dir=""):
    print("Making a random 2-fold split in the training samples with ratio %d:1"%(ratio))
    
    split_path = root_dir + "data_2_split/"
    
    training_sample_num = ecfp_tr.shape[0]
    one_part = int(training_sample_num/(ratio+1))
    samples_user_1 = ratio * one_part
    samples_user_2 = one_part
    print("Number of training samples user-1: %d"%(samples_user_1))
    print("Number of training samples user-2: %d"%(samples_user_2))
    
    shuffled_idx = np.array(range(training_sample_num))
    np.random.shuffle(shuffled_idx)
    user_1 = shuffled_idx[:samples_user_1]
    user_2 = shuffled_idx[-samples_user_2:]

    user_1_train_size = int(0.8*samples_user_1)
    user_2_train_size = int(0.8*samples_user_2)
    
    user_1_train = user_1[:user_1_train_size]
    user_1_test = user_1[user_1_train_size:]
    user_2_train = user_2[:user_2_train_size]
    user_2_test = user_2[user_2_train_size:]

    # T_total - number of targets
    T_total = ic50_tr.shape[1]
    ## select random target
    
    shuffled_labels = np.array(range(T_total))
    np.random.shuffle(shuffled_labels)
    T_half = int(T_total/2)
    user_1_labels = shuffled_labels[:T_half]
    user_2_labels = shuffled_labels[-T_half:]
    
    u1_ic50_tr = ic50_tr.tocsc()[:,user_1_labels].tocsr()
    u2_ic50_tr = ic50_tr.tocsc()[:,user_2_labels].tocsr()
    
    ## save user-1 data
    save_data(split_path + "0_train/", ecfp_tr[user_1_train],
            u1_ic50_tr[user_1_train])
    save_data(split_path + "0_test/", ecfp_tr[user_1_test],
            u1_ic50_tr[user_1_test])
    
    ## save user-2 data
    save_data(split_path + "1_train/", ecfp_tr[user_2_train],
            u2_ic50_tr[user_2_train])
    save_data(split_path + "1_test/", ecfp_tr[user_2_test],
            u2_ic50_tr[user_2_test])

def split_data_targets(num_targets, data_size, ecfp, ic50, seed=None,
        victim_target=False):
    """
    Return a subset of the passed data, with a certain number of targets (that
    get randomly selected).

    Parameters
    ----------
    num_targets : int
        Number of targets to select.
    data_size : int
        Number of samples to put into the returned dataset.
    ecfp : scipy.sparse.csr_matrix
        Training data.
    ic50 : scipy.sparse.csr_matrix
        Labels.
    seed : int
        The number with which to initialise the random number generator.
    """
    print("Get subset of passed data (size=%d), with %d targets" % (data_size,
        num_targets))
    if seed is not None:
        random.seed(seed)
    
    if victim_target:
        selected_targets = list(range(1404))
    else:
        max_targets = ic50.shape[1]
        # select targets between 1404-2808
        selected_targets = random.sample(population=range(1404, max_targets),
                k=num_targets)
    
    # training data should not match, re-initialize seed
    random.seed()
    max_samples = ecfp.shape[0]
    selected_samples = random.sample(population=range(max_samples),
            k=data_size)
    
    ## csr_matrix format is good for row-wise slicing (mtx[indices, :])
    ic50_s = ic50[selected_samples].tocsc()
    ## csc_matrix format is good for column-wise slicing (mtx[:, indices])
    ic50_s_t = ic50_s[:, selected_targets].tocsr()

    return ecfp[selected_samples], ic50_s_t

def array_in_list(arr, list_arr):
    """
    Returns indices of matches. A match is where the target array is the same
    as the ones in the list.

    Parameters
    ----------
    array : np.array
        The array we wish to match.
    list_arr : list
        The list of arrays where we search.

    Returns
    -------
    list
        List containing indices of matches.
    """
    match_indices = []
    for i, arr_2 in enumerate(list_arr):
        #print(arr==arr_2)
        if len(arr) == len(arr_2):
            if (arr==arr_2).all():
                match_indices.append(i)
    #print(match_indices)
    return match_indices

def get_split_points(v, max_value):
    """
    Returns an array of indices where v's value changes by 1. Where the change
    is greater (v[i+1] - v[i] > 1), then the index will appear n times in the
    result (given that n = v[i+1] - v[i]).

    Parameters
    ----------
    v : numpy.array
        The numpy array to examine.
    max_value : int
        Maximum possible value to appear in v.

    Returns
    -------
    numpy.array
        Array of indices where v's value changes by 1.
    """
    if len(v) in [0,1]:
        return np.array([0] * max_value)

    greatest_change = np.max(v[1:] - v[:-1])
    
    results = []
    first_change = v[0]
    if first_change != 0:
        res = np.repeat([0], first_change)
        results.append(res)
    for n in range(1,greatest_change+1):
        res = np.where(v[:-1]+n == v[1:])[0] + 1
        res = np.repeat(res, n)
        results.append(res)
    last_change = max_value - v[-1]
    if last_change != 0:
        res = np.repeat([len(v)], last_change) 
        results.append(res)
    if results != []:
        return np.sort(np.hstack(results))
    else:
        return []
 
def target_in_batch(batch, X_target, Y_target):
    """
    Returns True if target is present in batch.

    Parameters
    ----------
    batch : dict
        Batch where we have to search for target.
    X_target : scipy.sparse.csr_matrix
        Features of the target.
    Y_target : scipy.sparse.csr_matrix
        Label of the target.

    Returns
    -------
    Boolean
    """
    ## v (array): v[i] indicates which sample batch["x_data"] belongs to
    v = batch["x_ind"].numpy()[0]
    ## split_points: array of indices where v's value changes
    ## example: v = [0, 0, 0, 1, 1, 2] --> split_points = [3, 5]
    split_points = get_split_points(v, max_value=batch["batch_size"])
    ## v (array): v[i] indicates which sample batch["y_data"] belongs to
    v = batch["y_ind"].numpy()[0]
    ## split_points_label: array of indices where v's value changes
    split_points_label = get_split_points(v, max_value=batch["batch_size"])
    
    ## split data along corresponding split points
    splitted_data_idx = np.split(batch["x_ind"].numpy()[1], split_points)
    splitted_data = np.split(batch["x_data"].numpy(), split_points)
    splitted_label_idx = np.split(batch["y_ind"].numpy()[1], split_points_label)
    splitted_label = np.split(batch["y_data"].numpy(), split_points_label)
    
    ## match indices - the index in the batch where the target matches the
    ## batch's sample
    match_data_idx = array_in_list(X_target.indices, splitted_data_idx)
    match_data = array_in_list(X_target.data, splitted_data)

    match_label_idx = array_in_list(Y_target.indices, splitted_label_idx)
    match_label = array_in_list(Y_target.data, splitted_label)
    
    #intersection_1 = reduce(np.intersect1d, (match_data_idx, match_data))
    #intersection_2 = reduce(np.intersect1d, (match_label_idx, match_label))
    #if len(intersection_1) != 0 and len(intersection_2) != 0:
    #    print(intersection_1)
    #    print(intersection_2)

    ## get intersection of the match indices
    intersection = reduce(np.intersect1d, (match_data_idx, match_data, match_label_idx, match_label))

    ## if the intersection is not empty then the target is present in the batch
    return len(intersection) != 0


def get_random_sample(X_data, Y_data):
    """
    Return a random sample from the passed data.

    Parameters
    ----------
    X_data : scipy.sparse.csr_matrix 
        Features.
    Y_data : scipy.sparse.csr_matrix
        Labels.

    Returns
    -------
    scipy.sparse.csr_matrix
        Feature matrix of the random sample.
    scipy.sparse.csr_matrix
        Label matrix of the random sample.
    """
    rand_int = random.randrange(X_data.shape[0])
    print("Random sample index: ", rand_int)
    
    return X_data[rand_int], Y_data[rand_int]


def get_random_samples(X_data, Y_data, n):
    """
    Return n random samples from the passed data

    Parameters
    ----------
    X_data : scipy.sparse.csr_matrix 
        Features.
    Y_data : scipy.sparse.csr_matrix
        Labels.
    n : int
        Number of returned samples.

    Returns
    -------
    list 
        List of dictionaries (keys: "x"-data, "y"-labels).
    """
    rand_integers = random.sample(range(X_data.shape[0]), k=n)
    print("Random samples: ", rand_integers)

    res = []

    for idx in rand_integers:
        d = {}
        d["x"] = X_data[idx]
        d["y"] = Y_data[idx]
        res.append(d)

    return res


def fold_input(X, folded_size):
    """
    Return the folded version of X.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix 
        The input samples which we will fold.
    folded_size : int
        The new size of samples in X.

    Returns
    -------
    scipy.sparse.csr_matrix 
        The folded version of X.
    """
    num_cols = X.shape[1]
    num_folds = int(num_cols // folded_size)
    remaining_cols = int(num_cols % folded_size)
    #print("Number of folds is: ", num_folds)
    #print("Remaining cols: ", remaining_cols)
    # convert to csc format for efficient column-slicing
    X = X.tocsc()
    # begin folding
    new_X = X[:, :folded_size]
    for i in range(1, num_folds):
        next_fold = X[:,i*folded_size:(i+1)*folded_size]
        new_X += next_fold
    remaining_fold = X[:, -remaining_cols:]
    f_shape = remaining_fold.shape
    #print("Remaining fold shape: ", f_shape)
    row = np.array(range(f_shape[1]))
    col = np.array(range(f_shape[1]))
    data = np.ones(f_shape[1])
    fold_mul = sparse.csc_matrix((data, (row,col)), shape=(f_shape[1], folded_size))
    remaining_fold *= fold_mul
    # multiply remaining fold with new_X
    new_X += remaining_fold
    #print("New X shape: ", new_X.shape)
    new_X.data = np.ones(new_X.nnz)

    return new_X.tocsr()