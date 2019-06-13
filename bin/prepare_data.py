import sys
import numpy as np
import pickle as pkl
import os

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

        # Delete label from data
        x[:, 0] = 0

        # For each phoneme found in the sentence, get the sequence of MFCCs and the label

        for j in range(seq_train[i].shape[0]):
            data_tr += [x[seq_train[i][j][0]:seq_train[i][j][1]]]
            labels_tr += [targets_train[i][j]]

    # Return an array of arrays for the data, and an array of float for the labels
    return np.array(data_tr), np.array(labels_tr)


def prepare_data(fname_dtest=None, fname_dtrain=None, n_phn=None, verbose=False):
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
    usage = "Usage: python prepare_data.py 2 data/train13.pkl data/test13.pkl"

    if len(sys.argv) != 4:
        print(usage)
        sys.exit(1)

    nclasses = int(sys.argv[1])
    train_inputfile = sys.argv[2]
    test_inputfile = sys.argv[3]
    train_outfiles = [train_inputfile.replace(".pkl", "_" + str(i) + ".pkl") for i in range(nclasses)]
    test_outfiles = [test_inputfile.replace(".pkl", "_" + str(i) + ".pkl") for i in range(nclasses)]


    ########## data preparation ##########
    xtrain, ytrain, xtest, ytest, IPHN = prepare_data(fname_dtest=test_inputfile, fname_dtrain=train_inputfile,\
                                                n_phn=nclasses, verbose=False)

    classes = np.unique(ytrain)
    for i, ic in enumerate(classes):
        if not (os.path.exists(train_outfiles[i]) & os.path.exists(test_outfiles[i]) ):
            # At least one of the files is missing
            pkl.dump(xtrain[ytrain == ic], open(train_outfiles[i], "wb"))
            pkl.dump(xtest[ytest == ic], open(test_outfiles[i], "wb"))
        else:
            print("(skip) class data exist:", train_outfiles[i], test_outfiles[i], file=sys.stderr)

    sys.exit(0)