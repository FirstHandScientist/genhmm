import numpy as np
import os
import torch
from torch.utils.data import Dataset,DataLoader
import sys
import pickle as pkl
from parse import parse
import json
from functools import partial
import time


def data_read_parse(fname, dim_zero_padding=False):
    xtrain_ = pkl.load(open(fname, "rb"))

    if (isinstance(xtrain_, tuple) or isinstance(xtrain_, list)) and len(xtrain_) == 1:
        xtrain_ = xtrain_[0]

    if isinstance(xtrain_[0], list):
        xtrain_ = [np.array(x).T for x in xtrain_]

    if dim_zero_padding and xtrain_[0].shape[1] % 2 != 0:
        xtrain_ = [np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1) for x in xtrain_]


    return xtrain_


def test_norm_minmax():
    x = np.array([[1, 3], [2, 2]])
    min_ = x.min(0)
    max_ = x.max(0)
    xn = norm_minmax(x, min_=min_, max_=max_)
    assert ((xn == np.array([[0, 1], [1, 0]])).all())


def norm_minmax(x, min_=None, max_=None):
    return ((x - min_.reshape(1, -1)) / (max_.reshape(1, -1) - min_.reshape(1, -1)))


def normalize(xtrain, xtest):
    """Normalize training data set between 0 and 1. Perform the same scaling on the testing set."""
    f_min = np.vectorize(lambda x : np.min(x, axis=0), signature="()->(k)")
    f_max = np.vectorize(lambda x : np.max(x, axis=0), signature="()->(k)")
    min_tr = np.min(f_min(xtrain), axis=0)
    max_tr = np.max(f_max(xtrain), axis=0)

    # The first component is zeros and can create division by 0
    min_tr[0] = 0
    max_tr[0] = 1
    f_perform_normalize = np.vectorize(partial(norm_minmax, min_=min_tr, max_=max_tr), signature="()->()", otypes=[np.ndarray])
    return f_perform_normalize(xtrain), f_perform_normalize(xtest)


def to_device(mdl, use_gpu=False, Mul_gpu=False):
    if use_gpu and torch.cuda.is_available():
        if not Mul_gpu:
            # default case, only one gpu
            device = torch.device('cuda')
            mdl.device = device
            mdl.pushto(mdl.device)

        else:
            for i in range(4):
                try:
                    time.sleep(np.random.randint(20))
                    device = torch.device('cuda:{}'.format(int(get_freer_gpu())))
                    print("Try to push to device: {}".format(device), file=sys.stderr)
                    mdl.device = device
                    mdl.pushto(mdl.device)
                    break
                except:
                    # if push error (maybe memory overflow, try again)
                    print("Push to device cuda:{} fail, try again ...", file=sys.stderr)
                    continue
    return mdl


def accuracy_fun(data_file, mdl=None):
    try:
        X = data_read_parse(data_file)
    except:
        return "0/1"

    mode, _, iclass_str = parse("{}.{}_class{}.pkl", os.path.basename(data_file))
    # Get the length of all the sequences
    l = [xx.shape[0] for xx in X]
    # zero pad data for batch training

    true_class = parse("{}_class{}.pkl", os.path.basename(data_file))[1]
    out_list = [mdl.forward(x_i) for x_i in X]
    out = np.array(out_list).transpose()

    # the out here should be the shape: data_size * nclasses
    class_hat = np.argmax(out, axis=0) + 1
    istrue = class_hat == int(true_class)
    print(data_file, "Done ...", "{}/{}".format(str(istrue.sum()), str(istrue.shape[0])), file=sys.stderr)
    return "{}/{}".format(str(istrue.sum()), str(istrue.shape[0])), format_out_list(out_list)


def parse_out_list(out_list_str):
    return [list(map(np.float64, x.split(","))) for x in out_list_str.split(";")]


def format_out_list(out_list):
    return ";".join([",".join(list(map(str, o))) for o in out_list])


def accuracy_fun_torch(data_file, mdl=None, batch_size_=128):
    try:
        X = data_read_parse(data_file)
    except:
        return "0/1"

    # Get the length of all the sequences
    l = [xx.shape[0] for xx in X]
    if X[0].shape[1] % 2 != 0:
        X = [np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1) for x in X]

    # zero pad data for batch training
    max_len_ = max([xx.shape[0] for xx in X])
    x_padded = pad_data(X, max_len_)
    batchdata = DataLoader(dataset=TheDataset(x_padded,
                                              lengths=l,
                                              device=mdl.hmms[0].device),
                           batch_size=batch_size_, shuffle=True)

    true_class = parse("{}_class{}.pkl", os.path.basename(data_file))[1]
    out_list = [mdl.forward(x) for x in batchdata]
    out = torch.cat(out_list, dim=1)

    # the out here should be the shape: data_size * nclasses
    class_hat = torch.argmax(out, dim=0) + 1
    print(data_file, "Done ...", "{}".format(acc_str(class_hat, true_class)), file=sys.stderr)

    return acc_str(class_hat, true_class), format_out_list(out.cpu().numpy().T.tolist())


def save_model(mdl, fname=None):
    with open(fname, "wb") as handle:
        pkl.dump(mdl, handle)
    return 0


def load_model(fname):
    """Loads a model on CPU by default."""
    return pkl.load(open(fname, "rb"))


def acc_str(class_hat, class_true):
    istrue = class_hat == int(class_true)
    return "{}/{}".format(str(istrue.sum().cpu().numpy()), str(istrue.shape[0]))


def test_acc_str():
    class_hat = torch.FloatTensor([1, 2])
    class_true = 2
    assert( acc_str(class_hat, 1) == "1/2" )
    assert( acc_str(class_hat, 2) == "1/2" )

    class_hat = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert( acc_str(class_hat, 1) == "1/10")

    class_hat = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 1, 10])
    assert (acc_str(class_hat, 1) == "2/10")


def append_class(data_file, iclass):
    return data_file.replace(".pkl", "_class" + str(iclass)+".pkl")

def test_append_class():
    assert( "sdnfnsljdg/nlfsdflsd_class100.pkl" == append_class("sdnfnsljdg/nlfsdflsd.pkl", 100) )

def divide(res_int):
    return res_int[0] / res_int[1]

def parse_(res_str):
    return list(parse("{:d}/{:d}",res_str))

def test_parse():
    assert([1, 2] == parse_("1/2"))

class TheDataset(Dataset):
    """Wrapper for DataLoader input."""
    def __init__(self, xtrain,  lengths, ytrain=None, device='cpu'):
        self.data = [torch.FloatTensor(x).to(device) for x in xtrain]
        if not (ytrain is None):
            self.y = ytrain
            assert(ytrain.shape[0] == len(self.data))

        self.lengths = lengths
        max_len_ = self.data[0].shape[0]
        self.mask = [torch.cat((torch.ones(l, dtype=torch.uint8), \
                                torch.zeros(max_len_ - l, dtype=torch.uint8))).to(device) \
                     for l in self.lengths]
        self.len = len(self.data)

    def __getitem__(self, index):
        try:
            return (self.data[index], self.mask[index], self.y[index])
        except:
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
    return [np.concatenate((xx, np.zeros((length - xx.shape[0], d)))) for xx in x]

def test_pad_data():
    x = [np.ones((3, 2)), np.ones((5, 2))]
    max_len = max([xx.shape[0] for xx in x])
    y = pad_data(x, max_len)

    assert((y[0] == np.array([[1,1],[1,1],[1,1],[0,0],[0,0]])).all()
           and (y[1] == x[1]).all())

def norm_prob(x, axis=None):
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
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


if __name__ == "__main__":
    #test_acc_str()
    #test_norm_prob()
    #test_pad_data()
    test_append_class()
    test_parse()
    pass