# file test.py
import sys
sys.path.append('.')

import collections
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

def save_model(file_dir, tfile):
    """Save the trained models
    file_dir: the directory of model to be saved
    file name is set as "trained.pkl"
    Note: pickle does not support to pickle lambda function, so cloudpickle is used to save models
    """
    model_dir = os.path.join(file_dir, "trained.pkl")
    with open(model_dir, "wb") as file:
        pkl.dump(tfile, file, pkl.HIGHEST_PROTOCOL)

def load(file_dir, file_name=None):
    """load the trained models"""
    if file_name is None:
        model = os.path.join(file_dir, "trained.pkl")
    else:
        model = os.path.join(file_dir, "trained.pkl")

    with open(model, "rb") as file:
        return pkl.load(file)
    
class data_normalizer():
    """The normalization is defined this way such that the training data and testing data are normalized by the same min, max parameters. 
    Since they can be a bit different for training and testing data."""
    def __init__(self):
        self.min = None
        self.max = None
    
    def get_range(self, indata):
        indata = np.concatenate(indata)
        self.min = np.min(indata, axis=0)
        self.max = np.max(indata, axis=0)
        
    def normalize(self, indata):
        indata = np.concatenate(indata)
        return (indata - self.min) / (self.max - self.min)
    

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
    # number of phones to train, one hmm per phone
    ########## hyperparameters setting ##########
    n_phn = 2
    n_components = 3
    n_prob_components = 2

    models = collections.defaultdict(dict)

    ########## data preparation ##########
    xtrain, ytrain, xtest, ytest, IPHN = prepare_data(data_folder=data_folder,\
                                                fname_dtest=fname_dtest, fname_dtrain=fname_dtrain,\
                                                n_phn=n_phn, verbose=True)
    data_normalizer = data_normalizer()
    data_normalizer.get_range(indata=xtrain)

    limit = 20
    #################### train one genHMM per phone ####################
    for phn_idx in range(n_phn):
        print("[Begin to train Model for Phone {}]".format(IPHN[phn_idx]))
        models["phn{}".format(IPHN[phn_idx])]["phone"] = IPHN[phn_idx]
        models["phn{}".format(IPHN[phn_idx])]["model"] = GenHMM(n_components=n_components,
                                                          n_prob_components=n_prob_components,
                                                          n_iter=50,
                                                          em_skip=10, tol=0,
                                                          log_dir="results/genHMM/phn{}".format(IPHN[phn_idx]) )



        # Get data for class phn_idx
        sub_xtrain = xtrain[ytrain == IPHN[phn_idx]]
        sub_length = np.array([sub_xtrain[i].shape[0] for i in range(sub_xtrain.shape[0])])
        sub_xtrain_n = data_normalizer.normalize(sub_xtrain)
    
        
        models["phn{}".format(IPHN[phn_idx])]["model"].fit(sub_xtrain_n[:np.cumsum(sub_length[:limit])[-1]], sub_length[:limit])

        #### save model is now disabled due to the error of TypeError: can't pickle _thread.lock objects##################
    #     save_model(file_dir="results/genHMM/phn{}".format(phn_idx),
    #                tfile=models["phn{}".format(phn_idx)])

    # save_model(file_dir="results/genHMM", tfile=models)
        

    #################### testing the trained model classification performance ####################
    ############ load the models here, waiting for model saving and loading methods fixed #######
    
    # test the trained models
    limit = 20
    xtest_n = data_normalizer.normalize(xtest)
    length_test = np.array([xtest[i].shape[0] for i in range(xtest.shape[0])])
    preded_scores = []
    key_list = []
    for key, method in models.items():
        print("[Predicting samples under model, Idx:{}, Phone:{}]".format(key, method["phone"]))
        key_list.append(key)
        p_scores = method["model"].pred_score(xtest_n[:np.cumsum(length_test[:limit])[-1]], length_test[:limit])
        preded_scores.append(p_scores)
        
    idx_y_predition = np.array(preded_scores).argmax(axis=0)
    y_pred = []
    for idx in idx_y_predition:
        y_pred.append( models[key_list[idx]]["phone"] )
        
    y_pred = np.array(y_pred)
    total_pred_error = np.sum(y_pred != ytest[:limit]) / ytest[:limit].shape[0]
    print("[The prediction err: {}, Phones: {}]".format(total_pred_error, IPHN))
