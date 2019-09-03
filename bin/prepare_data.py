import sys
import numpy as np
import pickle as pkl
import os
from functools import partial
from parse import parse
import json
from gm_hmm.src.utils import read_classmap,write_classmap,flip,\
    phn61_to_phn39, remove_label, to_phoneme_level,getsubset,normalize


def get_phoneme_mapping(iphn, phn2int, n_taken=0):
    # Reverse the codebook for easier manipulation
    int2phn = flip(phn2int)
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



def prepare_data(fname_dtest=None, classmap_existing=None, fname_dtrain=None, n_phn=None,totclasses=None, verbose=False):
    # Read the datafiles
    te_DATA, te_keys, te_lengths, phn2int_61, te_PHN = pkl.load(open(fname_dtest, "rb"))
    tr_DATA, tr_keys, tr_lengths, tr_PHN = pkl.load(open(fname_dtrain, "rb"))

    if verbose:
        print("Data loaded from files.")

    # Take labels down to phoneme level
    data_tr, label_tr = to_phoneme_level(tr_DATA)
    data_te, label_te = to_phoneme_level(te_DATA)

    # Checkout table 3 at
    # https://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database
    # Or in the html file
    # for details
    phn2int = phn2int_61
    if totclasses == 39:
        f = partial(phn61_to_phn39, int2phn_61=flip(phn2int_61), data_folder=os.path.dirname(fname_dtest))
        label_tr, phn2int_39 = f(label_tr)
        label_te, _ = f(label_te, phn2int_39=phn2int_39)

        data_tr, label_tr = remove_label(data_tr, label_tr, phn2int_39)
        data_te, label_te = remove_label(data_te, label_te, phn2int_39)
        phn2int_39.pop('-', None)
        phn2int = phn2int_39

    # List the phoneme names already in the data folder.
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



if __name__ == "__main__":
    usage = "Build separate datasets for each family of phonemes.\n\"" \
            "Each data set contains the sequences of one phoneme.\n"\
            "Usage: python bin/prepare_data.py \"[nclasses]/[totclasses (61|39)]\" [training data] [testing data]\n"\
            "Example: python bin/prepare_data.py 2/61 data/train13.pkl data/test13.pkl"

    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    nclasses, totclasses = parse("{:d}/{:d}", sys.argv[1])
    train_inputfile = sys.argv[2]
    test_inputfile = sys.argv[3]
    # 
    train_outfiles = [train_inputfile.replace(".pkl", "_" + str(i+1) + ".pkl") for i in range(nclasses)]
    test_outfiles = [test_inputfile.replace(".pkl", "_" + str(i+1) + ".pkl") for i in range(nclasses)]
    data_folder = os.path.dirname(test_inputfile)

    classmap = read_classmap(data_folder)
    n_existing = len(classmap)

    if totclasses != 39 and totclasses != 61:
        print("(error)", "first argument must be [nclasses]/[61 or 39]", file=sys.stderr)
        print(usage, file=sys.stderr)
        sys.exit(1)

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
                                                n_phn=nclasses_fetch,
                                                classmap_existing=classmap,
                                                totclasses=totclasses,
                                                verbose=False)
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
