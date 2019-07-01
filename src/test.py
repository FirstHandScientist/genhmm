# test the batch forward, backward algorithm, and _compute_log_xi_sum
import os
import sys
sys.path.append("..")
from parse import parse
from scipy.special import logsumexp
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from hmmlearn import _hmmc
from _torch_hmmc import _forward, _backward, _compute_log_xi_sum



def batch_data(n_components):
    """ Generate a batch of framelogprob with zero paddings`
    Input:
    n_components: number of hmm states 

    OUTPUT: 

    data: Batch framelogprob (randomly generated) with zero paddings
          Shape: batch_size * max_sequence_length * feature vector length
    
    mask: The mask indicating which element of data is actually framelogprob, which element is padding
          Shape: batch_size * max_sequence_length
    """
    x = []
    # generate framelogprob 1, assume number of states is 4
    x.append(- np.abs(np.random.randn(3, n_components) ) )
    x.append(- np.abs(np.random.randn(5, n_components) ) )

    # assume do padding to 8 samples for every sequence

    length = 8
    d = x[0].shape[1]
    # generate xx here
    # mask should be included here
    data = np.array([np.concatenate((xx,np.zeros((length - xx.shape[0] + 1,d)))) for xx in x])
    mask = np.array( [xx.sum(axis=1) < 0 for xx in data ], dtype=np.uint8)
    
    return (data, mask)

# 1. batch data load
n_components = 4
startprob = np.abs(np.random.randn(n_components))
startprob = startprob/startprob.sum()
print("hmm start probability: {}".format(startprob) )

transmat = np.abs(np.random.randn(n_components, n_components))
transmat = transmat/transmat.sum(axis=0)
print("hmm transition probability: \n{}".format(transmat) )

data, mask = batch_data(n_components)
print("batch_framelogprob is: \n{}".format(data))
print("batch_mask is: \n{}".format(mask))

# 2. forward comparison

# 2.1 forward by hmmlearn method:
hmmlearn_fwdlattice = []
hmmlearn_logprob = []
for idx, framelogprob in enumerate(data):
    fwdlattice = np.zeros((mask[idx].sum(), n_components))
    _hmmc._forward(int(mask[idx].sum()), \
                   int(n_components), \
                   np.log(startprob), \
                   np.log(transmat), \
                   framelogprob[mask[idx]>0], \
                   fwdlattice)
    hmmlearn_fwdlattice.append(fwdlattice)
    hmmlearn_logprob.append(logsumexp(fwdlattice[-1]))
print("hmmlearn forward lattice: \n {}\n".format(hmmlearn_fwdlattice))
print("hmmlearn logporb: {}\n".format(hmmlearn_logprob))
# 2.2 batch forward in PyTorch

torch_logprob, torch_fwdlattice = _forward(int(data.shape[1]), \
                               int(n_components), \
                               torch.from_numpy(np.log(startprob)), \
                               torch.from_numpy(np.log(transmat)), \
                               torch.from_numpy(data), \
                               torch.from_numpy(mask))
print("torch batch forward: \n {}\n".format(torch_fwdlattice))
print("torch batch logprob: {}\n".format(torch_logprob))

# 3. backward comparison

hmmlearn_bwdlattice = []
for idx, framelogprob in enumerate(data):
    bwdlattice = np.zeros((mask[idx].sum(), n_components))
    _hmmc._backward(int(mask[idx].sum()), \
                   int(n_components), \
                   np.log(startprob), \
                   np.log(transmat), \
                   framelogprob[mask[idx]>0], \
                   bwdlattice)
    hmmlearn_bwdlattice.append(bwdlattice)
print("hmmlearn backward lattice: \n{}".format(hmmlearn_bwdlattice))

# 3.2 batch backward in PyTorch

torch_bwdlattice = _backward(int(data.shape[1]), \
                             int(n_components), \
                             torch.from_numpy(np.log(startprob)), \
                             torch.from_numpy(np.log(transmat)), \
                             torch.from_numpy(data), \
                             torch.from_numpy(mask))
print("torch batch backward: \n{}".format(torch_bwdlattice))



# 4. log_xi_sum

## 4.1 compute the log_xi_sum by hmmlearn implementation
hmmlearn_log_xi_sum = []
for idx, framelogprob in enumerate(data):
    log_xi_sum = np.full((n_components, n_components), -np.inf)
    _hmmc._compute_log_xi_sum(int(mask[idx].sum()), \
                              int(n_components), \
                              hmmlearn_fwdlattice[idx], \
                              np.log(transmat), \
                              hmmlearn_bwdlattice[idx], \
                              framelogprob, \
                              log_xi_sum)
    hmmlearn_log_xi_sum.append(log_xi_sum)
print("hmmlearn log_xi_sum lattice:\n{}".format(hmmlearn_log_xi_sum))

## 4.2 compute teh log_xi_sum by batch version computation of log_xi_sum
log_xi_sum = torch.ones(2, n_components, n_components)*float('-inf')
torch_log_xi_sum = _compute_log_xi_sum(int(data.shape[1]), \
                                       int(n_components), \
                                       torch_fwdlattice, \
                                       torch.from_numpy(np.log(transmat)), \
                                       torch_bwdlattice, \
                                       torch.from_numpy(data), \
                                       log_xi_sum, \
                                       torch_logprob, \
                                       torch.from_numpy(mask))
print("torch log_xi_sum lattice: {}".format(torch_log_xi_sum))
#Todo: assert hmmlearn_logprob == torch_log_xi_sum.numpy()
