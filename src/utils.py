import numpy as np
import os
import torch
from torch.utils.data import Dataset


class TheDataset(Dataset):
    """Wrapper for DataLoader input."""
    def __init__(self, xtrain, lengths, device='cpu'):
        self.data = [torch.FloatTensor(x).to(device) for x in xtrain]
        self.lengths = lengths
        max_len_ = self.data[0].shape[0]
        self.mask = [torch.cat((torch.ones(l, dtype=torch.uint8), \
                                torch.zeros(max_len_ - l, dtype=torch.uint8))).to(device) \
                     for l in self.lengths]
        self.len = len(self.data)

    def __getitem__(self, index):
        return (self.data[index], self.mask[index])

    def __len__(self):
        return self.len


def pad_data(x, length):
    """ Add zeros at the end of all sequences in to get sequences of lengths `length`
    Input:  x : list, all of sequences of variable length to pad
            length : integer, common target length of sequences.
    output: list,  all input sequences zero-padded.
    """

    d = x[0].shape[1]
    return [np.concatenate((xx, np.zeros((length - xx.shape[0] + 1, d)))) for xx in x]


def norm_prob(x,axis=None):
    coef_ = x.sum(axis)
    if axis==0:
        coef_ = coef_.reshape(1,-1)
    elif axis==1:
        coef_ = coef_.reshape(-1, 1)

    return x / np.repeat(coef_, x.shape[axis], axis=axis)

def test_norm_prob():
    x = np.array([[1, 2], [4, 5]])
    xn = norm_prob(x, axis=1)
    assert(np.all(xn == np.array([[1/(1+2), 2/(1+2)], [4/(4+5), 5/(4+5)]])))
    xn = norm_prob(x, axis=0)
    assert (np.all(xn == np.array([[1 / (1 + 4), 2 / (5 + 2)], [4 / (4 + 1), 5 / (2 + 5)]])))


def step_learning_rate_decay(init_lr, global_step, minimum,
                             anneal_rate=0.98,
                             anneal_interval=1):
    rate = init_lr * anneal_rate ** (global_step // anneal_interval)
    if rate < minimum:
        rate = minimum
    return rate

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


if __name__ == "__main__":
    test_norm_prob()
