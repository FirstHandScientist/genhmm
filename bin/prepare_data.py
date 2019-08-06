import sys
import numpy as np
import pickle as pkl
import os
from functools import partial
import json

def getsubset(data, label, iphn):
    # concat data
    # find subset
    idx = np.in1d(label, iphn)
    return data[idx], label[idx]


def find_change_loc(x):
    dx = np.diff(x)
    # Make a clean vector to delimit phonemes
    change_locations = np.array([0] + (1 + np.argwhere(dx != 0)).reshape(-1).tolist() + [x.shape[0]])
    # Make an array of size n_phoneme_in_sentence x 2, containing begining and end of each phoneme in a sentence

    fmt_interv = np.array([[change_locations[i-1], change_locations[i]]\
                                 for i in range(1, change_locations.shape[0]) ])
    return fmt_interv, x[change_locations[:-1]]


def test_find_change_loc():
    l = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,2,2,2,2,2,2,3])
    out, out2 = find_change_loc(l)
    assert((out2 == np.array([1,0,2,3])).all())
    assert((out == np.array([[0,  7], [7, 13],[13, 19],[19, 20]])).all())

    l = np.array([1, 1, 0, 0, 2, 2])
    out, out2 = find_change_loc(l)
    assert((out2 == np.array([1, 0, 2])).all())
    assert((out == np.array([[0, 2], [2, 4], [4, 6]])).all())


def to_phoneme_level(DATA):
    n_sequences = len(DATA)

    seq_train = [0 for _ in range(n_sequences)]
    targets_train = [0 for _ in range(n_sequences)]
    data_tr = []
    labels_tr = []

    # For all sentences
    for i, x in enumerate(DATA):
        seq_train[i], targets_train[i] = find_change_loc(x[:,0])

        # Delete label from data
        x[:, 0] = 0

        # For each phoneme found in the sentence, get the sequence of MFCCs and the label
        for j in range(seq_train[i].shape[0]):
            data_tr += [x[seq_train[i][j][0]:seq_train[i][j][1]]]
            labels_tr += [targets_train[i][j]]

    # Return an array of arrays for the data, and an array of float for the labels
    return np.array(data_tr), np.array(labels_tr)

def get_phoneme_mapping(iphn, phn2int, n_taken=0):
    # Reverse the codebook for easier manipulation
    int2phn = {y: x for x, y in phn2int.items()}
    # Define a sub-dictionary of the picked phonemes
    class2phn = {j+n_taken: int2phn[i] for j, i in enumerate(iphn)}
    class2int = {j+n_taken: i for j, i in enumerate(iphn)}
    return class2phn, class2int


def test_get_phoneme_mapping():
    phn2int = {"a": 0, "b": 2, "c": 1, "d": 3}
    iphn = np.array([1, 2])
    assert(get_phoneme_mapping(iphn, phn2int) == {0:"c", 1:"b"})
    iphn = np.array([2, 1])
    assert(get_phoneme_mapping(iphn, phn2int) == {0:"b", 1:"c"})


def prepare_data(fname_dtest=None, classmap_existing=None, fname_dtrain=None, n_phn=None, verbose=False):
    # Read the datafiles
    te_DATA, te_keys, te_lengths, phn2int, te_PHN = pkl.load(open(fname_dtest, "rb"))
    tr_DATA, tr_keys, tr_lengths, tr_PHN = pkl.load(open(fname_dtrain, "rb"))

    if verbose:
        print("Data loaded from files.")

    # Take labels down to phoneme level
    data_tr, label_tr = to_phoneme_level(tr_DATA)
    data_te, label_te = to_phoneme_level(te_DATA)


    # list the phoneme names already in the data folder.
    taken = [v for k, v in classmap_existing.items()]

    # Deduce the available phonemes
    available_phn = [v for k, v in phn2int.items() if not k in taken]

    # Pick new random phonemes
    iphn = np.random.permutation(available_phn)[:n_phn]

    # Find the phonemes in the dataset
    xtrain, ytrain = getsubset(data_tr, label_tr, iphn)
    xtest, ytest = getsubset(data_te, label_te, iphn)


    class2phn, class2int = get_phoneme_mapping(iphn, phn2int, n_taken=len(taken))

    return xtrain, ytrain, xtest, ytest, class2phn, class2int


def read_classmap(folder):
    fname = os.path.join(folder, "class_map.json")
    if os.path.isfile(fname):
        with open(fname, "r") as f:
            return json.load(f)
    else:
        return {}


def write_classmap(class2phn, folder):
    """ Write dictionary to a JSON file."""
    with open(os.path.join(folder, "class_map.json"), "w") as outfile:
        out_str = json.dumps(class2phn, indent=2)
        print("Classes are: \n" + out_str, file=sys.stderr)
        outfile.write(out_str+"\n")
    return 0


def test_norm_minmax():
    x = np.array([[1, 3], [2, 2]])
    min_ = x.min(0)
    max_ = x.max(0)
    xn = norm_minmax(x, min_=min_, max_=max_)
    assert ((xn == np.array([[0, 1 ], [1,0]])).all())


def norm_minmax(x, min_=None, max_=None):
    return ((x - min_.reshape(1, -1)) / (max_.reshape(1, -1) - min_.reshape(1, -1)))

def norm_musigma(x, mu_=None, sigma_=None):
    return ((x - mu_) / sigma_)


def test_normalize():
    xtrain = np.array( [np.random.rand(np.random.randint(20),14) for _ in range(30)])
    xtest = np.array([np.random.rand(np.random.randint(20), 14) for _ in range(10)])
    tmp_tr = np.concatenate(xtrain.tolist(), axis=0)
    tmp_te = np.concatenate(xtest.tolist(), axis=0)

    tmp_tr_n = (tmp_tr - tmp_tr.mean(0)[None, :]) / tmp_tr.std(0)[None, :]
    tmp_te_n = (tmp_te - tmp_tr.mean(0)[None, :]) / tmp_tr.std(0)[None, :]

    xtrain_n, xtest_n = normalize(xtrain, xtest)
    assert ((np.concatenate(xtrain_n.tolist(), axis=0) == tmp_tr_n).all())
    assert ((np.concatenate(xtest_n.tolist(), axis=0) == tmp_te_n).all())

def normalize(xtrain, xtest):
    """Normalize training data set such that mean = 0 and std = 1 for training set.
    Perform the same scaling on the testing set.
    It looks a bit complex because I want to keep the sequences of variables lengths."""

    tmp_tr = np.concatenate(xtrain.tolist(), axis=0)
    mu_tr = tmp_tr.mean(0)[None, :]
    sigma_tr = tmp_tr.std(0)[None, :]

    # The first component is zeros and can create division by 0
    sigma_tr[sigma_tr == 0] = 1

    f_perform_normalize = np.vectorize(
        partial(norm_musigma, mu_=mu_tr, sigma_=sigma_tr),
        signature="()->()", otypes=[np.ndarray])
    return f_perform_normalize(xtrain), f_perform_normalize(xtest)


if __name__ == "__main__":
    usage = "Build separate datasets for each family of phonemes.\n\"" \
            "Each data set contains the sequences of one phoneme.\n"\
            "Usage: python bin/prepare_data.py [nclasses] [training data] [testing data]\n"\
            "Example: python bin/prepare_data.py 2 data/train13.pkl data/test13.pkl"

    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    nclasses = int(sys.argv[1])
    train_inputfile = sys.argv[2]
    test_inputfile = sys.argv[3]
    # 
    train_outfiles = [train_inputfile.replace(".pkl", "_" + str(i+1) + ".pkl") for i in range(nclasses)]
    test_outfiles = [test_inputfile.replace(".pkl", "_" + str(i+1) + ".pkl") for i in range(nclasses)]
    data_folder = os.path.dirname(test_inputfile)

    classmap = read_classmap(data_folder)
    n_existing = len(classmap)

    # Print the number of classes which already exist
    if n_existing > 0:
        print("(info)", n_existing, "classes already exist.", file=sys.stderr)

    # We request less classes than there already are, we skip and check that the files are indeed present
    if n_existing >= nclasses:
        print("(skip)", nclasses, "classes already exist.", file=sys.stderr)
        assert(all([os.path.isfile(x) for x in train_outfiles + test_outfiles]))
        sys.exit(0)


    # Number of classes left to fetch
    nclasses_fetch = nclasses - n_existing
    print("(info)", nclasses_fetch, "classes to fetch.")

    # Now {x,y}{train,test} only contain newly picked phonemes (not present in classmap)
    xtrain, ytrain, xtest, ytest, class2phn, class2int = prepare_data(fname_dtest=test_inputfile, fname_dtrain=train_inputfile,\
                                                n_phn=nclasses_fetch, classmap_existing=classmap, verbose=False)
    # normalization 
    xtrain, xtest = normalize(xtrain, xtest)

    classmap = {**classmap, **class2phn}

    # Assert length (If we add an already existing phoneme,
    # the dictionary size will not be len(classmap) + len(class2phn)
    assert (len(classmap) == nclasses)



    # Create only the classes that are left
    for i, ic in class2int.items():
        assert(not os.path.isfile(train_outfiles[i]))
        assert(not os.path.isfile(test_outfiles[i]))
        xtrain_c = xtrain[ytrain == ic]
        xtest_c = xtest[ytest == ic]
        
        pkl.dump(xtrain_c, open(train_outfiles[i], "wb"))
        pkl.dump(xtest_c, open(test_outfiles[i], "wb"))

    # Write the mapping class number <=> phoneme
    write_classmap(classmap, os.path.dirname(test_inputfile))

    sys.exit(0)
