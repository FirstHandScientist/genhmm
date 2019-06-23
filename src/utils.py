# file to for utilities
import os
import sys
sys.path.append("..")
import pickle as pkl
import numpy as np

import torch
from functools import partial

def pad_data(x,length):
    d = x[0].shape[1]
    return [np.concatenate((xx,np.zeros((length - xx.shape[0] + 1,d)))) for xx in x]
 
def normalize(a, axis=None):
    """Normalizes the input array so that it sums to 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis, keepdim=True)
    a_sum[a_sum==0] = 1
    a /= a_sum


def accuracy_fun(data_file, mdl=None):
    data = pkl.load(open(data_file, "rb"))
    max_len_ = max([x.shape[0] for x in data])
    X = pad_data(data,length=max_len_)
    true_class = parse("{}_{}.pkl", os.path.basename(data_file))[1]

    out = np.array(list(map(mdl.forward, X)))
    class_hat = np.argmax(out, axis=1) + 1
    istrue = class_hat == int(true_class)
    return "{}/{}".format(str(istrue.sum()), str(istrue.shape[0]))


def append_class(data_file, iclass):
    return data_file.replace(".pkl", "_" + str(iclass)+".pkl")


def divide(res_int):
    return res_int[0] / res_int[1]


def parse_(res_str):
    res_str_split = res_str.split("/")
    res_int = [int(x) for x in res_str_split]
    return res_int


def print_results(mdl_file, results, data_files):
    # Print class by class accuracy
    for res, data_f in zip(results, data_files):
        print(mdl_file, res[0], data_f[0].astype("<U"), divide(parse_(data_f[0].astype("<U"))), sep='\t', file=sys.stdout)
        print(mdl_file, res[1], data_f[1].astype("<U"), divide(parse_(data_f[1].astype("<U"))), sep='\t', file=sys.stdout)

    # Print total accuracy
    res = np.concatenate([np.array([parse_(r[0].astype("<U")), parse_(r[1].astype("<U"))]).T for r in data_files],axis=1)
    tr_res = res[:, ::2]
    te_res = res[:, 1::2]
    tr_res_str = str(tr_res[0].sum()/tr_res[1].sum())
    te_res_str = str(te_res[0].sum()/te_res[1].sum())
    print("Acc:", mdl_file, tr_res_str, te_res_str, sep='\t', file=sys.stdout)





def getsubset(data, label, iphn):
    # concat data
    # find subset
    idx = np.in1d(label, iphn)
    return data[idx], label[idx]


def to_phoneme_level(DATA):
    n_sequences = len(DATA)

    seq_train = [0 for _ in range(n_sequences)]
    targets_train = [0 for _ in range(n_sequences)]
    data_tr = []
    labels_tr = []

    # For all sentences
    for i, x in enumerate(DATA):
        # Note: first column contains labels. Find when the label changes, it indicates a change of phoneme in the sentence
        dx = np.diff(x[:, 0])

        # Make a clean vector to delimit phonemes
        change_locations = np.array([0] + (1 + np.argwhere(dx != 0)).reshape(-1).tolist() + [x.shape[0]-1])

        # Make an array of size n_phoneme_in_sentence x 2, containing begining and end of each phoneme in a sentence
        seq_train[i] = np.array([[change_locations[i-1], change_locations[i]]\
                                 for i in range(1, change_locations.shape[0]) ])#np.diff(change_locations)

        # Save an instance of labels
        targets_train[i] = x[change_locations[:-1], 0]

        # Delete label from data
        x[:, 0] = 0

        # For each phoneme found in the sentence, get the sequence of MFCCs and the label
        for j in range(seq_train[i].shape[0]):
            data_tr += [x[seq_train[i][j][0]:seq_train[i][j][1]]]
            labels_tr += [targets_train[i][j]]

    # Return an array of arrays for the data, and an array of float for the labels
    return np.array(data_tr), np.array(labels_tr)


def prepare_data(fname_dtest=None, fname_dtrain=None, n_phn=None, verbose=False):
    # Read the datafiles
    te_DATA, te_keys, te_lengths, codebook, te_PHN = pkl.load(open(fname_dtest, "rb"))
    tr_DATA, tr_keys, tr_lengths, tr_PHN = pkl.load(open(fname_dtrain, "rb"))
    if verbose:
        print("Data loaded from files.")

    # Take labels down to phoneme level
    data_tr, label_tr = to_phoneme_level(tr_DATA)
    data_te, label_te = to_phoneme_level(te_DATA)


    # Pick n random phonemes
    iphn = np.random.randint(len(codebook), size=n_phn)

    xtrain, ytrain = getsubset(data_tr, label_tr, iphn)
    xtest, ytest = getsubset(data_te, label_te, iphn)

    return xtrain, ytrain, xtest, ytest, iphn


def norm_minmax(x, min_=None, max_=None):
    return ( (x - min_.reshape(1, -1)) / (max_.reshape(1, -1) - min_.reshape(1, -1)))


def normalize(xtrain, xtest):
    f_min = np.vectorize(lambda x : np.min(x, axis=0), signature="()->(k)")
    f_max = np.vectorize(lambda x : np.max(x, axis=0), signature="()->(k)")
    min_tr = np.min(f_min(xtrain), axis=0)
    max_tr = np.max(f_max(xtrain), axis=0)

    # The first component is zeros and can create division by 0
    min_tr[0] = 0
    max_tr[0] = 1
    f_perform_normalize = np.vectorize(partial(norm_minmax, min_=min_tr, max_=max_tr), signature="()->()",otypes=[np.ndarray])
    return f_perform_normalize(xtrain), f_perform_normalize(xtest)




def save_model(mdl, fname=None):
    torch.save(wrapper(mdl), fname)
    return 0

def load_model(fname):
    savable = torch.load(fname)
    return savable.userdata




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


if __name__ == "__main__":
    test_norm_prob()
