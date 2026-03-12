from torch.utils.data import Sampler
import functools
import numpy as np
import torch
import random
import math

class CustomSampler(Sampler):
    """
    A custom created sampler class that returns batches of indices so that the
    target sample's indices don't get seperated.

    ...
    
    Attributes
    ----------
    indices : list
        Complete list of data indices.
    target_sample_indices : list
        A list of the target indices that need to stay in the same batch.
    batch_size : int
        Number of samples in a batch.
    """
    def __init__(self, indices, target_sample_indices, batch_size):
        self.indices = indices
        self.target_sample_indices = target_sample_indices
        self.batch_size = batch_size

        self._shuffle()
        
    def _shuffle(self):
        num_of_samples = len(self.indices)

        self.indices = np.setdiff1d(self.indices, self.target_sample_indices)
        
        ## shuffle indices
        np.random.shuffle(self.indices)

        ## select random index
        ## so that the reinserted target indices don't get selected into
        ## different batches
        target_size = len(self.target_sample_indices)
        offset = random.randrange(self.batch_size-target_size)
        mul = random.randrange(int(num_of_samples/self.batch_size))
        idx = self.batch_size*mul + offset

        ## reinsert target indices at the previously computed random index
        self.indices = self.indices.tolist()
        self.indices[idx:idx] = self.target_sample_indices

    def __iter__(self):
        self._shuffle()
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class PositiveSampler(Sampler):
    """
    A custom created sampler class that returns batches containing the target
    samples.

    ...
    
    Attributes
    ----------
    indices : list
        Complete list of data indices.
    target_sample_indices : list
        A list of the target indices that need to stay in the same batch.
    batch_size : int
        Number of samples in a batch.
    """
    def __init__(self, indices, target_sample_indices, batch_size):
        self.indices = list(indices)
        self.target_sample_indices = target_sample_indices
        self.batch_size = batch_size

        self._shuffle()
        
    def _shuffle(self):
        num_of_samples = len(self.indices)

        ## shuffle indices
        np.random.shuffle(self.indices)
        
        ## insert target sample indices into every batch
        num_of_batches = math.ceil(num_of_samples/self.batch_size)
        for b_i in range(num_of_batches):
            idx = b_i * self.batch_size
            num_targets = len(self.target_sample_indices)
            self.indices[idx:idx+num_targets] = self.target_sample_indices


    def __iter__(self):
        self._shuffle()
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)