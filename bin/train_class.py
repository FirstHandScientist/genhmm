import os
import sys
from parse import parse
import pickle as pkl
from src.genHMM import GenHMM
import numpy as np
import torch

class wrapper(torch.nn.Module):
    def __init__(self, mdl):
        super(wrapper, self).__init__()
        self.userdata = mdl

def save_model(mdl, fname=None):
    torch.save(wrapper(mdl), fname)
    return 0

def load_model(fname):
    savable = torch.load(fname)
    return savable.userdata


if __name__ == "__main__":
    usage = "python bin/train_class.py data/train13.pkl models/epoch0_class1.mdlc"
    if len(sys.argv) != 3:
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
        mdl = pkl.load(open(out_mdl.replace("epoch" + epoch_str, "epoch" + str(int(epoch_str)-1)), "rb"))


    mdl.fit(X, lengths=l)

    save_model(mdl,fname=out_mdl)

    #pkl.dump(mdl, open(out_mdl, "wb"))
    sys.exit(0)
