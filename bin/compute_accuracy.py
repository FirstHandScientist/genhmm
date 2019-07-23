import sys
import os

from functools import partial
import pickle as pkl
# need to import GaussianHMMclassifier to load it via pkl
from gm_hmm.src.ref_hmm import GaussianHMMclassifier
import numpy as np
from parse import parse
from gm_hmm.src.utils import divide, acc_str, append_class, parse_

import torch
from torch.utils.data import DataLoader


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

    # Parse argument
    mdl_file = sys.argv[1]
    training_data_file = sys.argv[2]
    testing_data_file = sys.argv[3]

    # Load Model
    with open(mdl_file, "rb") as handle:
        mdl = pkl.load(handle)


    # Prepare for computation of results
    nclasses = len(mdl.hmms)

    # Builds an array of string containing the train and test data sets for each class
    # size: nclass x 2 (train, test)
    data_files = np.array([[append_class(training_data_file, iclass+1), append_class(testing_data_file, iclass+1)]
                   for iclass in range(nclasses)])

    # Define a function for this particular HMMclassifier model
    f = partial(accuracy_fun, mdl=mdl)

    # force the output type of f to an object, f_v: (str) class data file -> (object) Model accuracy
    f_v = np.vectorize(f, otypes=["O"])
    
    # Call the f_v function on all class data files and transform the result from an array of object to an array of string .
    results = f_v(data_files).astype('|S1024')
    
    print_results(mdl_file, data_files, results)
    sys.exit(0)

