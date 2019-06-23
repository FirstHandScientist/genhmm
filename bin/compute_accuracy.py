import sys
sys.path.append("..")

import os
from src.genHMM import load_model
from functools import partial
import pickle as pkl
import numpy as np
from parse import parse
from src.utils import append_class, divide, parse_, print_results, load_model


# def pad_data(x,length):
#     d = x[0].shape[1]
    
#     return [torch.cat((torch.FloatTensor(xx),torch.zeros((length - xx.shape[0] + 1,d))),0) for xx in x]

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

