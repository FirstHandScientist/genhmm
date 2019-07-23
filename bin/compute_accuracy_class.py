import os
import sys
from parse import parse
import pickle as pkl
from gm_hmm.src.utils import append_class,accuracy_fun
from functools import partial

if __name__ == "__main__":
    usage = "bin/compute_accuracy_class.py epoch2_class1 13feats/data/train.13.pkl 13feats/data/test.13.pkl 13feats/models"
    if len(sys.argv) != 5 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage, file=sys.stderr)
        sys.exit(1)

    epoch, iclass = parse("epoch{:d}_class{:d}", sys.argv[1])

    training_data_file = sys.argv[2]
    testing_data_file = sys.argv[3]
    models_dir = sys.argv[4]
    mdl_file = os.path.join(models_dir, "epoch{}.mdl".format(epoch))

    # Load Model
    with open(mdl_file, "rb") as handle:
        mdl = pkl.load(handle)

    # Prepare for computation of results
    nclasses = len(mdl.hmms)

    # Builds an array of string containing the train and test data sets for each class
    # size: nclass x 2 (train, test)
    data_files = [append_class(training_data_file, iclass + 1), append_class(testing_data_file, iclass + 1)]
    f = partial(accuracy_fun, mdl=mdl)

    results = list(map(f, data_files))

    print("epoch: {} class: {} train: {} test: {}".format(epoch, iclass + 1, results[0], results[1]))
