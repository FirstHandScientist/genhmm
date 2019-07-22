import os
import sys
from parse import parse
import pickle as pkl
import json
import numpy as np
import time
from hmmlearn import hmm
from gm_hmm.src.ref_hmm import ConvgMonitor
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state




def create_random_gmm(n_mix, n_features, covariance_type):
    g = GaussianMixture(n_mix, covariance_type=covariance_type)
    g.means_ = np.random.randn(n_mix, n_features) * 0.01
    g.covars_ = make_covar_matrix(covariance_type, n_mix, n_features)
    g.weights_ = np.ones(n_mix) / n_mix
    return g

def make_covar_matrix(covariance_type, n_components, n_features,
                      random_state=None):
    mincv = 0.1
    prng = check_random_state(random_state)
    if covariance_type == 'spherical':
        return (mincv + mincv * prng.random_sample((n_components,))) ** 2
    elif covariance_type == 'tied':
        return (make_spd_matrix(n_features)
                + mincv * np.eye(n_features))
    elif covariance_type == 'diag':
        return (mincv + mincv *
                prng.random_sample((n_components, n_features))) ** 2
    elif covariance_type == 'full':
        return np.array([
            (make_spd_matrix(n_features, random_state=prng)
             + mincv * np.eye(n_features))
            for x in range(n_components)
        ])

if __name__ == "__main__":
    usage = "python bin/train_class_gmm.py data/train13.pkl models/epoch1_class1.mdlc param.json"
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
    xtrain = [x for x in xtrain_]
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
        # init GaussianHMM model
        mdl = hmm.GaussianHMM(n_components=options["Net"]["n_states"], \
                              covariance_type="full", tol=-np.inf, verbose=True)
        # mdl = hmm.GaussianHMM(n_components=options["Net"]["n_states"], \
        #                       covariance_type="full", tol=-np.inf)
        mdl.monitor_ = ConvgMonitor(mdl.tol, mdl.n_iter, mdl.verbose)
        # param setting
        mdl.startprob_ = np.ones(mdl.n_components) /mdl.n_components
        tmp_transmit = np.ones(mdl.n_components, mdl.n_components) + \
                       np.random.randn(mdl.n_components, mdl.n_components) * 0.01
        
        mdl.transmat_ = tmp_transmit / tmp_transmit.sum(axis=1)
        # mdl.gmms = [ create_random_gmm(options["Net"]["n_prob_components"], \
        #                                options["Net"]["net_D"], \
        #                                "diag") for _ in range(mdl.n_components)]
        #mdl._init(xtrain, l)

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


