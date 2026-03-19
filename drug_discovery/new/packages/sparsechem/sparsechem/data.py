from torch.utils.data import Dataset
import torch
import scipy.sparse
import numpy as np
import random
import scipy.sparse as sparse

class SparseDataset(Dataset):
    def __init__(self, x, y):
        '''
        Args:
            X (sparse matrix): input [n_sampes, features_in]
            Y (sparse matrix): output [n_samples, features_out]
        '''
        self.x = x.tocsr(copy=False).astype('float32')
        self.y = y.tocsr(copy=False).astype('float32')

        assert self.x.shape[0]==self.y.shape[0], f"Input has {self.x.shape[0]} rows, output has {self.y.shape[0]} rows."

    def __len__(self):
        return(self.x.shape[0])

    @property
    def input_size(self):
        return self.x.shape[1]

    @property
    def output_size(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        xi = self.x[idx,:]
        yi = self.y[idx,:]

        return {
            "x_ind":  xi.indices,
            "x_data": xi.data,
            "y_ind":  yi.indices,
            "y_data": yi.data,
        }

    def batch_to_x(self, batch, dev):
        """Takes 'xind' and 'x_data' from batch and converts them into a sparse tensor.
        Args:
            batch  batch
            dev    device to send the tensor to
        """
        return torch.sparse_coo_tensor(
                batch["x_ind"].to(dev),
                batch["x_data"].to(dev),
                size=[batch["batch_size"], self.x.shape[1]])

    def replace_sample_with(self, X_replacement, y_replacement, index):
        """
        Replace data on index with X_replacement and y_replacement.

        Parameters
        ----------
        X_replacement : scipy.sparse.csr_matrix
            The replacement sample's data.
        y_replacement : scipy.sparse.csr_matrix
            The replacement sample's label.
        """
        self.x = self.x.tolil()
        self.x[index, :] = X_replacement.toarray()
        self.x = self.x.tocsr()

        self.y = self.y.tolil()
        self.y[index, :] = y_replacement.toarray()
        self.y = self.y.tocsr()


    def replace_random_sample_with(self, X_replacement, y_replacement):
        """
        Replace a random sample from the dataset to the sample in the
        parameter.

        Parameters
        ----------
        X_replacement : scipy.sparse.csr_matrix
            The replacement sample's data.
        y_replacement : scipy.sparse.csr_matrix
            The replacement sample's label.
        """
        inside_dataset = False
        idx = None
        for i in range(self.x.shape[0]):
            if (self.x[i] != X_replacement).nnz==0 and (self.y[i] != y_replacement).nnz==0:
                inside_dataset = True
                idx = i
        if inside_dataset:
            print("Not replacing anything. Sample already on index %d" % idx)
        else:
            index = random.randrange(len(self))
            print("Replacing sample on index %d." % index)
            
            #print("Old sample:")
            #print("\tx indices: \n", self.x[index].indices)
            #print("\ty indices: \n", self.y[index].indices) 

            self.replace_sample_with(X_replacement, y_replacement, index)
            
            #print("New sample:")
            #print("\tx indices: \n", self.x[index].indices)
            #print("\ty indices: \n", self.y[index].indices) 


class MappingDataset(Dataset):
    def __init__(self, x_ind, x_data, y, mapping=None):
        """
        Dataset that creates a mapping for features of x (0...N_feat-1).
        """
        pass

def sparse_collate(batch):
    x_ind  = [b["x_ind"]  for b in batch]
    x_data = [b["x_data"] for b in batch]
    y_ind  = [b["y_ind"]  for b in batch]
    y_data = [b["y_data"] for b in batch]

    ## x matrix
    xrow = np.repeat(np.arange(len(x_ind)), [len(i) for i in x_ind])
    xcol = np.concatenate(x_ind)
    xv   = np.concatenate(x_data)

    ## y matrix
    yrow  = np.repeat(np.arange(len(y_ind)), [len(i) for i in y_ind])
    ycol  = np.concatenate(y_ind).astype(np.int64)

    return {
        "x_ind":  torch.LongTensor([xrow, xcol]),
        "x_data": torch.from_numpy(xv),
        "y_ind":  torch.LongTensor([yrow, ycol]),
        "y_data": torch.from_numpy(np.concatenate(y_data)),
        "batch_size": len(batch),
    }
