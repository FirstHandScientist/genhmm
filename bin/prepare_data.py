import sys
sys.path.append("..")

import numpy as np
import pickle as pkl
import os
from functools import partial
from src.utilities import getsubset, to_phoneme_level, prepare_data




if __name__ == "__main__":
    usage = "Usage: python bin/prepare_data.py [nclasses] [training data] [testing data]"

    if len(sys.argv) != 4 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    nclasses = int(sys.argv[1])
    train_inputfile = sys.argv[2]
    test_inputfile = sys.argv[3]
    train_outfiles = [train_inputfile.replace(".pkl", "_" + str(i+1) + ".pkl") for i in range(nclasses)]
    test_outfiles = [test_inputfile.replace(".pkl", "_" + str(i+1) + ".pkl") for i in range(nclasses)]


    ########## data preparation ##########
    xtrain, ytrain, xtest, ytest, IPHN = prepare_data(fname_dtest=test_inputfile, fname_dtrain=train_inputfile,\
                                                n_phn=nclasses, verbose=False)

    classes = np.unique(ytrain)
    for i, ic in enumerate(classes):
        if not (os.path.exists(train_outfiles[i]) & os.path.exists(test_outfiles[i]) ):
            # At least one of the files is missing
            xtrain_c = xtrain[ytrain == ic]
            xtest_c = xtest[ytest == ic]

            xtrain_cn, xtest_cn = normalize(xtrain_c, xtest_c)

            pkl.dump(xtrain_cn, open(train_outfiles[i], "wb"))
            pkl.dump(xtest_cn, open(test_outfiles[i], "wb"))

        else:
            print("(skip) class data exist:", train_outfiles[i], test_outfiles[i], file=sys.stderr)

    sys.exit(0)
