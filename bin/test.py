# file test.py
import pickle as pkl
import sys
import os
import numpy as np

from src.genHMM import GenHMM




def getsubset(data, label, iphn):
    # concat data
    # find subset
    idx = np.in1d(label, iphn)
    return data[idx], label[idx]


def to_phoneme_level(DATA):
    n_sequences = len(DATA)

    seq_train = [0 for _ in range(n_sequences)]
    targets_train = [0 for _ in range(n_sequences)]
    data_tr = []
    labels_tr = []

    # For all sentences
    for i, x in enumerate(DATA):
        # Note: first column contains labels. Find when the label changes, it indicates a change of phoneme in the sentence
        dx = np.diff(x[:, 0])

        # Make a clean vector to delimit phonemes
        change_locations = np.array([0] + (1 + np.argwhere(dx != 0)).reshape(-1).tolist() + [x.shape[0]-1])

        # Make an array of size n_phoneme_in_sentence x 2, containing begining and end of each phoneme in a sentence
        seq_train[i] = np.array([[change_locations[i-1], change_locations[i]]\
                                 for i in range(1, change_locations.shape[0]) ])#np.diff(change_locations)

        # Save an instance of labels
        targets_train[i] = x[change_locations[:-1], 0]

        # For each phoneme found in the sentence, get the sequence of MFCCs and the label

        for j in range(seq_train[i].shape[0]):
            data_tr += [x[seq_train[i][j][0]:seq_train[i][j][1]]]
            labels_tr += [targets_train[i][j]]

    # Return an array of arrays for the data, and an array of float for the labels
    return np.array(data_tr), np.array(labels_tr)


def prepare_data(data_folder=None, fname_dtest=None, fname_dtrain=None, n_phn=None, verbose=False):
    # Read the datafiles
    te_DATA, te_keys, te_lengths, codebook, te_PHN = pkl.load(open(fname_dtest, "rb"))
    tr_DATA, tr_keys, tr_lengths, tr_PHN = pkl.load(open(fname_dtrain, "rb"))
    if verbose:
        print("Data loaded from files.")

    # Take labels down to phoneme level
    data_tr, label_tr = to_phoneme_level(tr_DATA)
    data_te, label_te = to_phoneme_level(te_DATA)


    # Pick n random phonemes
    iphn = np.random.randint(len(codebook), size=n_phn)

    xtrain, ytrain = getsubset(data_tr, label_tr, iphn)
    xtest, ytest = getsubset(data_te, label_te, iphn)

    return xtrain, ytrain, xtest, ytest, iphn


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

    # Use a dataset of two phones for simplicity
    n_phn = 2
    xtrain, ytrain, xtest, ytest, IPHN = prepare_data(data_folder=data_folder,\
                                                fname_dtest=fname_dtest, fname_dtrain=fname_dtrain,\
                                                n_phn=n_phn, verbose=True)

    n_components = 3
    n_prob_components = 2
    hmm1 = GenHMM(n_components=n_components, n_prob_components=n_prob_components, hparams=hparams)

    # Get data for class 1
    data1 = xtrain[ytrain == IPHN[0]]
    length1 = np.array([data1[i].shape[0] for i in range(data1.shape[0])])
    data1 = np.concatenate(data1)
    limit = 10
    hmm1.fit(data1[:np.cumsum(length1[:limit])[-1]], length1[:limit])
    llh = hmm1.llh(xtrain[:100])


    # hmm1.fit(xtrain[ytrain == IPHN[0]])

    print("")