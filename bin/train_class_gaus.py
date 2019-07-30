import os
import sys
from parse import parse
import pickle as pkl
import json
import numpy as np
import time

from gm_hmm.src.ref_hmm import Gaussian_HMM, GMM_HMM, ConvgMonitor
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state


if __name__ == "__main__":
    usage = "python bin/train_class_gaus.py exp/gaus/39feats/data/train.13.pkl exp/gaus/39feats/models/epoch1_class1.mdlc param.json"
    if len(sys.argv) < 3 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    # Parse
    train_inputfile = sys.argv[1]
    out_mdl = sys.argv[2]

    # Test for a third parameter
    try:
        param_file = sys.argv[3]
    except IndexError:
        param_file = "default.json"


    epoch_str, iclass_str = parse('epoch{}_class{}.mdlc',os.path.basename(out_mdl))
    train_class_inputfile = train_inputfile.replace(".pkl", "_{}.pkl".format(iclass_str))

    #  Load data
    
    xtrain_ = pkl.load(open(train_class_inputfile, "rb"))
    xtrain = [x[:,1:] for x in xtrain_]
    xtrain = np.concatenate(xtrain, axis=0)
    #xtrain = xtrain[:100]
    # Get the length of all the sequences
    l = [x.shape[0] for x in xtrain_]

    # load the parameters
    with open(param_file) as f_in:
        options = json.load(f_in)

    # adoptive to set number of states
    options["Net"]["n_states"] = np.clip(int(np.floor(np.mean(l)/2)),
                                         options["Train"]["n_states_min"],
                                         options["Train"]["n_states_max"])

    #  Load or create model
    if epoch_str == '1':
        # init GaussianHMM model or GMM_HMM model by disable/comment one and enable another model. For GMM_HMM, we are now just maintaining diag type of covariance.
        mdl = GMM_HMM(n_components=options["Net"]["n_states"], \
                      n_mix= 2, #options["Net"]["n_prob_components"], \
                      covariance_type="diag", tol=-np.inf, \
                      init_params="stwmc", params="", verbose=True)
        # mdl = Gaussian_HMM(n_components=options["Net"]["n_states"], \
        #                    covariance_type="full", tol=-np.inf, verbose=True)
        mdl.monitor_ = ConvgMonitor(mdl.tol, mdl.n_iter, mdl.verbose)
        # param setting
        # There is self._init(X, lengths=lengths) in fit method, which would initialize the following parameters according to input data. So The following initialization would be overwritten (thus can be commented out) if self._init is executed in fit.
        mdl.startprob_ = np.ones(mdl.n_components) /mdl.n_components
        tmp_transmit = np.ones(mdl.n_components, mdl.n_components) + \
                       np.random.randn(mdl.n_components, mdl.n_components) * 0.01
        
        mdl.transmat_ = tmp_transmit / tmp_transmit.sum(axis=1)

    else:
        # Load previous model
        mdl = pkl.load(open(out_mdl.replace("epoch" + epoch_str, "epoch" + str(int(epoch_str)-1)), "rb"))

    mdl.iepoch = epoch_str
    mdl.iclass = iclass_str
    
    print("epoch:{}\tclass:{}\t.".format(epoch_str,iclass_str), file=sys.stdout)
    
    # zero pad data for batch training
    # niter counts the number of em steps before saving a model checkpoint
    niter = options["Train"]["niter"]
    
    # add number of training data in model
    # mdl.number_training_data = len(xtrain)
    
    mdl.n_iter = niter
    
    mdl.fit(xtrain, lengths=l)

    # Push back to cpu for compatibility when GPU unavailable.
    with open(out_mdl, "wb") as handle:
        pkl.dump(mdl, handle)
    sys.exit(0)


