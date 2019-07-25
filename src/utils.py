import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import pickle as pkl
from parse import parse


def accuracy_fun(data_file, mdl=None):
    X = pkl.load(open(data_file, "rb"))
    # Get the length of all the sequences
    l = [xx.shape[0] for xx in X]
    # zero pad data for batch training

    true_class = parse("{}_{}.pkl", os.path.basename(data_file))[1]
    out_list = [mdl.forward(x_i) for x_i in X]
    out = np.array(out_list).transpose()

    # the out here should be the shape: data_size * nclasses
    class_hat = np.argmax(out, axis=0) + 1
    istrue = class_hat == int(true_class)
    print(data_file, "Done ...", "{}/{}".format(str(istrue.sum()), str(istrue.shape[0])), file=sys.stderr)
    return "{}/{}".format(str(istrue.sum()), str(istrue.shape[0]))

def accuracy_fun_torch(data_file, mdl=None):
    X = pkl.load(open(data_file, "rb"))
    # Get the length of all the sequences
    l = [xx.shape[0] for xx in X]
    # zero pad data for batch training
    max_len_ = max([xx.shape[0] for xx in X])
    x_padded = pad_data(X, max_len_)
    batchdata = DataLoader(dataset=TheDataset(x_padded,
                                              lengths=l,
                                              device=mdl.hmms[0].device),
                           batch_size=512, shuffle=True)
    
    true_class = parse("{}_{}.pkl", os.path.basename(data_file))[1]
    out_list = [mdl.forward(x) for x in batchdata]
    out = torch.cat(out_list, dim=1)

    # the out here should be the shape: data_size * nclasses
    class_hat = torch.argmax(out, dim=0) + 1
    print(data_file, "Done ...", "{}".format(acc_str(class_hat, true_class)), file=sys.stderr)

    return acc_str(class_hat, true_class)

def acc_str(class_hat,class_true):
    istrue = class_hat == int(class_true)
    return "{}/{}".format(str(istrue.sum().cpu().numpy()), str(istrue.shape[0]))


def test_acc_str():
    class_hat = torch.FloatTensor([1, 2])
    class_true = 2
    assert( acc_str(class_hat, 1) == "1/2" )
    assert( acc_str(class_hat, 2) == "1/2" )

    class_hat = torch.FloatTensor([1,2,3,4,5,6,7,8,9,10])
    assert( acc_str(class_hat, 1) == "1/10")

    class_hat = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 1, 10])
    assert (acc_str(class_hat, 1) == "2/10")

def append_class(data_file, iclass):
    return data_file.replace(".pkl", "_" + str(iclass)+".pkl")


def divide(res_int):
    return res_int[0] / res_int[1]



def parse_(res_str):
    res_str_split = res_str.split("/")
    res_int = [int(x) for x in res_str_split]
    return res_int

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
    test_acc_str()
    test_norm_prob()
