from parse import parse
import sys
import argparse
import os
import pickle as pkl
from gm_hmm.src.utils import read_classmap,to_phoneme_level,flip, phn61_to_phn39,remove_label, getsubset, normalize
from functools import partial

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a test dataset according to an existing class map.")
    parser.add_argument('-input', metavar="<Input dataset to split>", type=str)
    parser.add_argument('-classmap', metavar="<class_map.json file>", type=str)
    parser.add_argument('-totclass', metavar="<Total number of classes>", type=int)
    args = parser.parse_args()

    fname_dtest = args.input
    mode, nfeats, ntype, snr = parse("{}.{}.{}.{:d}db.pkl", os.path.basename(fname_dtest))
    fname_dtrain = os.path.join(os.path.dirname(fname_dtest), "train.{}.pkl".format(nfeats))

    cmap = read_classmap(os.path.dirname(args.classmap))
    te_DATA, te_keys, te_lengths, phn2int_61, te_PHN = pkl.load(open(fname_dtest, "rb"))

    tr_DATA, tr_keys, tr_lengths, tr_PHN = pkl.load(open(fname_dtrain, "rb"))

    data_te, label_te = to_phoneme_level(te_DATA)
    data_tr, label_tr = to_phoneme_level(tr_DATA)
    phn2int = phn2int_61
    if args.totclass == 39:
        f = partial(phn61_to_phn39, int2phn_61=flip(phn2int_61), data_folder=os.path.dirname(fname_dtest))
        label_tr, phn2int_39 = f(label_tr)
        label_te, _ = f(label_te, phn2int_39=phn2int_39)

        data_te, label_te = remove_label(data_te, label_te, phn2int_39)
        data_te, label_te = remove_label(data_te, label_te, phn2int_39)

        phn2int_39.pop('-', None)
        phn2int = phn2int_39

    iphn = [phn2int[v] for k, v in cmap.items()]

    xtrain, ytrain = getsubset(data_tr, label_tr, iphn)
    xtest, ytest = getsubset(data_te, label_te, iphn)
    xtrain, xtest = normalize(xtrain, xtest)

    # Assert length (If we add an already existing phoneme,
    # the dictionary size will not be len(classmap) + len(class2phn)

    test_outfiles = [fname_dtest.replace(".pkl", "_" + str(i+1) + ".pkl") for i in range(len(iphn))]

    # Create only the classes that are left
    for i, ic in zip(map(int, cmap.keys()), iphn):
        assert (not os.path.isfile(test_outfiles[i]))
        xtest_c = xtest[ytest == ic]
        pkl.dump(xtest_c, open(test_outfiles[i], "wb"))

    
    sys.exit(0)
