# file test.py
import pickle as pkl
import sys
import os
import numpy as np

from src.genHMM import GenHMM

def prepare_data(data_folder=None, fname_dtest=None, fname_dtrain=None, n_phn=None, verbose=False):
    te_DATA, te_keys, te_lengths, codebook, te_PHN = pkl.load(open(fname_dtest, "rb"))
    tr_DATA, tr_keys, tr_lengths, tr_PHN = pkl.load(open(fname_dtrain, "rb"))
    if verbose:
        print("Data loaded from files.")

    # Pick n random phoneme
    iphn = np.random.randint(len(codebook), size=n_phn)

    # Prepare training data
    X = np.concatenate(tuple(tr_DATA), axis=0)
    d = find_phones(X, iphn)
    ytrain = d[:, 0]
    # Keep label as a dummy value to have even dimensions
    xtrain = d[:, :]
    del d

    # Prepare test data
    X = np.concatenate(tuple(te_DATA), axis=0)
    d = find_phones(X, iphn)
    ytest = d[:, 0]
    # Keep label as a dummy value to have even dimensions
    xtest = d[:, :]
    del d

    return xtrain, ytrain, xtest, ytest, iphn
    pass


def find_phones(X=None, iphn=None):
    return X[np.concatenate(tuple([(X[:, 0] == iphn[i]).reshape(-1, 1) for i in range(iphn.shape[0])]), axis=1).sum(1) == 1, :]


if __name__ == "__main__":
    
    usage = "Usage: python bin/test.py [data folder] [test pkl file] [train pkl file] [params .json file]\n\n\
            Example: python bin/test.py data/ test13.pkl train13.pkl hparams/test.json\n"

    if len(sys.argv) != 5:
        print(usage)
        sys.exit(1)

    data_folder = sys.argv[1]
    fname_dtest = os.path.join(data_folder, sys.argv[2])
    fname_dtrain = os.path.join(data_folder, sys.argv[3])
    hparams = sys.argv[4]

    n_phn = 2
    xtrain, ytrain, xtest, ytest, IPHN = prepare_data(data_folder=data_folder, \
                                                fname_dtest=fname_dtest, fname_dtrain=fname_dtrain,\
                                                n_phn=n_phn, verbose=True)


    n_components = 3
    n_prob_components = 2
    hmm1 = GenHMM(n_components=n_components, n_prob_components=n_prob_components, hparams=hparams)
    llh=hmm1.llh(xtrain[:100])
    #hmm1.fit(xtrain[ytrain == IPHN[0]])

    print("")