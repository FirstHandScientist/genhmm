import sys
import os

from src.genHMM import load_model
from functools import partial
import pickle as pkl
import numpy as np
from parse import parse


def accuracy_fun(data_file, mdl=None):
    X = pkl.load(open(data_file, "rb"))
    true_class = parse("{}_{}.pkl", os.path.basename(data_file))[1]

    out = np.array([mdl.forward(x) for x in X[:100]])
    class_hat = np.argmax(out, axis=1) + 1
    istrue = class_hat == int(true_class)
    return "{}/{}".format(str(istrue.sum()), str(istrue.shape[0]))


def append_class(data_file, iclass):
    return data_file.replace(".pkl", "_" + str(iclass)+".pkl")

def divide(res_str):
    res_str_split = res_str.split("/")
    res_int = [int(x) for x in res_str_split]
    return res_int[0] / res_int[1]


def print_results(mdl_file, results, data_files):
    for res, data_f in zip(results, data_files):
        print(mdl_file, res[0], data_f[0].astype("<U"), divide(data_f[0].astype("<U")), sep='\t', file=sys.stdout)
        print(mdl_file, res[1], data_f[1].astype("<U"), divide(data_f[1].astype("<U")), sep='\t', file=sys.stdout)


if __name__ == "__main__":
    usage = "Usage: python bin/compute_accuracy.py [mdl file] [ training and testing data .pkl files]\n" \
            "Example: python bin/compute_accuracy.py models/epoch1.mdl data/train13.pkl data/test13.pkl" \

    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage, file=sys.stderr)
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

