import random
from functools import reduce
import scipy.io
import numpy as np
from os import path
from tqdm import tqdm


def load_data(rel_path):
    X = scipy.io.mmread(path.join(rel_path, "T11_x.mtx")).tocsr()
    Y = scipy.io.mmread(path.join(rel_path, "T10_y.mtx")).tocsr()
    return X, Y


def load_partner_data(rel_path, partners_idx=[]):
    if not partners_idx:
        return
    X, Y = load_data(rel_path)
    partner_data = []
    for partner in partners_idx:
        rows_idx = np.genfromtxt(path.join(rel_path, "partner_%d_xy_row_index_map.csv" % partner),
                                 dtype=int,
                                 delimiter=',',
                                 skip_header=1,
                                 usecols=[1])
        cols_idx = np.genfromtxt(path.join(rel_path, "partner_%d_y_col_index_map.csv" % partner),
                                 dtype=int,
                                 delimiter=',',
                                 skip_header=1,
                                 usecols=[1])
        X_partner = X[rows_idx]
        Y_partner = Y[rows_idx].tocsc()[:, cols_idx].tocsr()

        print("Loaded %d. partner with %d samples and %d targets." % (partner, len(rows_idx), len(cols_idx)))

        partner_data.append((X_partner, Y_partner))

    return partner_data


def load_member_non_member_data(rel_path, partners_idx=[]):
    if not partners_idx:
        return
    X, Y = load_data(rel_path)

    print("Creating list of member indices...")
    member_indices = []
    for partner in tqdm(partners_idx):
        rows_idx = np.genfromtxt(path.join(rel_path, "partner_%d_xy_row_index_map.csv" % partner),
                                 dtype=int,
                                 delimiter=',',
                                 skip_header=1,
                                 usecols=[1])
        member_indices.extend(rows_idx)
    member_indices = set(member_indices)
    print("Creating list of non-member indices...")
    non_member_indices = list(range(X.shape[0]))
    non_member_indices = [i for i in tqdm(non_member_indices) if i not in member_indices]
    member_indices = list(member_indices)

    x_member, y_member = X[member_indices], Y[member_indices]
    x_non_member, y_non_member = X[non_member_indices], Y[non_member_indices]

    return x_member, y_member, x_non_member, y_non_member


def get_random_sample(X_data):
    rand_int = random.randrange(X_data.shape[0])
    return X_data[rand_int]


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
        # print(arr==arr_2)
        if len(arr) == len(arr_2):
            if (arr == arr_2).all():
                match_indices.append(i)
    # print(match_indices)
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
    if len(v) in [0, 1]:
        return np.array([0] * max_value)

    greatest_change = np.max(v[1:] - v[:-1])

    results = []
    first_change = v[0]
    if first_change != 0:
        res = np.repeat([0], first_change)
        results.append(res)
    for n in range(1, greatest_change + 1):
        res = np.where(v[:-1] + n == v[1:])[0] + 1
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


def target_in_batch(batch, X_target):
    """
    Returns True if target is present in batch.

    Parameters
    ----------
    batch : dict
        Batch where we have to search for target.
    X_target : scipy.sparse.csr_matrix
        Features of the target.

    Returns
    -------
    Boolean
    """
    ## v (array): v[i] indicates which sample batch["x_data"] belongs to
    v = batch["x_ind"].numpy()[0]
    ## split_points: array of indices where v's value changes
    ## example: v = [0, 0, 0, 1, 1, 2] --> split_points = [3, 5]
    split_points = get_split_points(v, max_value=batch["batch_size"])

    ## split data along corresponding split points
    splitted_data_idx = np.split(batch["x_ind"].numpy()[1], split_points)
    splitted_data = np.split(batch["x_data"].numpy(), split_points)


    ## match indices - the index in the batch where the target matches the
    ## batch's sample
    match_data_idx = array_in_list(X_target.indices, splitted_data_idx)
    match_data = array_in_list(X_target.data, splitted_data)

    ## get intersection of the match indices
    intersection = reduce(np.intersect1d, (match_data_idx, match_data))

    ## if the intersection is not empty then the target is present in the batch
    return len(intersection) != 0

