import os
import sys
from parse import parse
import pickle as pkl
import json
import numpy as np
import time

from gm_hmm.src.ref_hmm import Gaussian_HMM, GMM_HMM, ConvgMonitor
from gm_hmm.src.utils import data_read_parse, load_model,save_model

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

    epoch_str, iclass_str = parse('epoch{}_class{}.mdlc', os.path.basename(out_mdl))
    train_class_inputfile = train_inputfile.replace(".pkl", "_class{}.pkl".format(iclass_str))

    #   Load data
    # Must output a list of arrays
    xtrain_ = data_read_parse(train_class_inputfile)
    xtrain = np.concatenate(xtrain_, axis=0)
    iremove = np.argwhere(xtrain.std(0) == 0)
    if iremove.shape[0] > 0:
        xtrain = np.delete(xtrain, iremove, axis=1)

    #   Get the length of all the sequence
    l = [x.shape[0] for x in xtrain_]
    
    #   Load the parameters
    with open(param_file) as f_in:
        options = json.load(f_in)

    #  Load or create model
    if epoch_str == '1':
        # init GaussianHMM model or GMM_HMM model by disable/comment one and enable another model. For GMM_HMM, we are now just maintaining diag type of covariance.
        mdl = GMM_HMM(n_components=options["GMM"]["n_states"], \
                      n_mix=options["GMM"]["n_prob_components"], \
                      covariance_type="diag", tol=-np.inf, \
                      init_params="stwmc", params="stwmc", verbose=True)

        # mdl = Gaussian_HMM(n_components=options["Net"]["n_states"], \
        #                    covariance_type="full", tol=-np.inf, verbose=True)
        mdl.monitor_ = ConvgMonitor(mdl.tol, mdl.n_iter, mdl.verbose)

    else:
        # Load previous model
        mdl = load_model(out_mdl.replace("epoch" + epoch_str, "epoch" + str(int(epoch_str)-1)))

    mdl.iepoch = epoch_str
    mdl.iclass = iclass_str
    mdl.train_data_fname = train_class_inputfile

    print("epoch:{}\tclass:{}\t.".format(epoch_str, iclass_str), file=sys.stdout)
    
    #  Zero pad data for batch training
    #  Niter counts the number of em steps before saving a model checkpoint
    niter = options["Train"]["niter"]
    
    #  Add number of training data in model
    #  Mdl.number_training_data = len(xtrain)
    
    mdl.n_iter = niter
    mdl.fit(xtrain, lengths=l)

    # Push back to cpu for compatibility when GPU unavailable.
    save_model(mdl, fname=out_mdl)
    sys.exit(0)
