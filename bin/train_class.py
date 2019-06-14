import os
import sys
from parse import parse
import pickle as pkl
from src.genHMM import GenHMM, save_model, load_model
import numpy as np
import torch


if __name__ == "__main__":
    usage = "python bin/train_class.py data/train13.pkl models/epoch1_class1.mdlc"
    if len(sys.argv) != 3 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    # Parse
    train_inputfile = sys.argv[1]
    out_mdl = sys.argv[2]

    epoch_str, iclass_str = parse('epoch{}_class{}.mdlc',os.path.basename(out_mdl))
    train_class_inputfile = train_inputfile.replace(".pkl", "_{}.pkl".format(iclass_str))

    #  Load data
    xtrain = pkl.load(open(train_class_inputfile, "rb"))
    xtrain = xtrain[:10]

    # Reshape data
    l = [x.shape[0] for x in xtrain]
    X = np.concatenate(xtrain)

    #  Load or create model
    if epoch_str == '1':
        #  Create model
        options = dict(n_components=2, n_prob_components=2,
                       n_iter=3000,
                       em_skip=20, tol=0)

        mdl = GenHMM(**options)

    else:
        # Load previous model
        mdl = load_model(out_mdl.replace("epoch" + epoch_str, "epoch" + str(int(epoch_str)-1)))

    mdl.fit(X, lengths=l)
    save_model(mdl, fname=out_mdl)
    sys.exit(0)
