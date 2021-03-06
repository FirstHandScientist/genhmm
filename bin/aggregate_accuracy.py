import sys
import os
from functools import partial
import pickle as pkl
# need to import GaussianHMMclassifier to load it via pkl
from gm_hmm.src.ref_hmm import GaussianHMMclassifier
import numpy as np
from parse import parse
from gm_hmm.src.utils import divide, acc_str, append_class, parse_, accuracy_fun

import torch
from torch.utils.data import DataLoader


def print_results(mdl_file, data_files, results):
    epoch = parse("epoch{}.mdl", os.path.basename(mdl_file))[0]
    # Print class by class accuracy
    for res, data_f in zip(results, data_files):
        true_class = parse("{}_{}.pkl", os.path.basename(data_f[0]))[1]

        print("epoch:", epoch, "class:", true_class, mdl_file, data_f[0].astype("<U"), res[0],
              divide(parse_(res[0].astype("<U"))), sep='\t', file=sys.stdout)
        print("epoch:", epoch, "class:", true_class, mdl_file, data_f[1].astype("<U"), res[1],
              divide(parse_(res[1].astype("<U"))), sep='\t', file=sys.stdout)

    # Print total accuracy
    res = np.concatenate([np.array([parse_(r[0].astype("<U")), parse_(r[1].astype("<U"))]).T for r in results], axis=1)
    tr_res = res[:, ::2]
    te_res = res[:, 1::2]
    tr_res_str = str(tr_res[0].sum() / tr_res[1].sum())
    te_res_str = str(te_res[0].sum() / te_res[1].sum())
    print("epoch:", epoch, "Acc:", mdl_file, tr_res_str, te_res_str, sep='\t', file=sys.stdout)


if __name__ == "__main__":
    usage = "Usage: python bin/aggregate_accuracy.py [train dataset] [test dataset] [list of .accc files]\n" \
            "Example: python bin/aggregate_accuracy.py python bin/aggregate_accuracy.py" \
                        "13feats/data/train.13.pkl 13feats/data/test.13.pkl"\
                        " 13feats/models/epoch2_class1.accc 13feats/models/epoch2_class2.accc"


    if len(sys.argv) < 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage, file=sys.stdout)
        sys.exit(1)
    training_data_file = sys.argv[1]
    testing_data_file = sys.argv[2]
    accc_files = sys.argv[3:]

    models_dir, epoch, _ = parse("{}/epoch{}_{}",accc_files[0])
    nclasses = len(accc_files)

    # Parse argument

    # rebuild information
    mdl_file = os.path.join(models_dir,"epoch{}.mdl".format(epoch))
    data_files = np.array([[append_class(training_data_file, iclass+1), append_class(testing_data_file, iclass+1)]
                   for iclass in range(nclasses)])

    # Read results
    def file2str(fname):
        with open(fname,"r") as f:
            line = f.read().strip().split("\n")[0]
        _, train_res, test_res = parse("{}accc train: {} test: {}", line)
        return train_res, test_res

    results = np.array([list(file2str(x)) for x in accc_files])

    print_results(mdl_file, data_files, results)
    sys.exit(0)

