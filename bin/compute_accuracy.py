import sys
sys.path.append("..")

import os
from src.genHMM import load_model
from functools import partial
import pickle as pkl
import numpy as np
from parse import parse
from train_class import pad_data, TheDataset
from torch.utils.data import DataLoader

def accuracy_fun(data_file, mdl=None):
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
    
    out = np.concatenate(list(map(mdl.forward, batchdata)), axis=1)

    # the out here should be the shape: data_size * nclasses
    class_hat = np.argmax(out, axis=0) + 1
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


def print_results(mdl_file, data_files, results):
    epoch=parse("epoch{}.mdl",os.path.basename(mdl_file))[0]
    # Print class by class accuracy
    for res, data_f in zip(results, data_files):
        true_class = parse("{}_{}.pkl", os.path.basename(data_f[0]))[1]
        
        print("epoch:",epoch, "class:",true_class, mdl_file, data_f[0].astype("<U"), res[0], divide(parse_(res[0].astype("<U"))), sep='\t', file=sys.stdout)
        print("epoch:",epoch, "class:",true_class, mdl_file, data_f[1].astype("<U"), res[1], divide(parse_(res[1].astype("<U"))), sep='\t', file=sys.stdout)

    # Print total accuracy
    res = np.concatenate([np.array([parse_(r[0].astype("<U")), parse_(r[1].astype("<U"))]).T for r in results],axis=1)
    tr_res = res[:, ::2]
    te_res = res[:, 1::2]
    tr_res_str = str(tr_res[0].sum()/tr_res[1].sum())
    te_res_str = str(te_res[0].sum()/te_res[1].sum())
    print("epoch:", epoch, "Acc:", mdl_file, tr_res_str, te_res_str, sep='\t', file=sys.stdout)


if __name__ == "__main__":
    usage = "Usage: python bin/compute_accuracy.py [mdl file] [ training and testing data .pkl files]\n" \
            "Example: python bin/compute_accuracy.py models/epoch1.mdl data/train13.pkl data/test13.pkl" \

    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage, file=sys.stdout)
        sys.exit(1)

    mdl_file = sys.argv[1]
    training_data_file = sys.argv[2]
    testing_data_file = sys.argv[3]

    mdl = load_model(mdl_file)
    nclasses = len(mdl.hmms)

    data_files = np.array([[append_class(training_data_file, iclass+1), append_class(testing_data_file, iclass+1)]
                   for iclass in range(nclasses)])

    f = partial(accuracy_fun, mdl=mdl)
    f_v = np.vectorize(f, otypes=["O"])
    results = f_v(data_files).astype('|S1024')

    print_results(mdl_file, data_files, results)
    sys.exit(0)

