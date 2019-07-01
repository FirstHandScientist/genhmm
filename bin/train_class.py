import os
import sys
sys.path.append("..")

from parse import parse
import pickle as pkl
from src.genHMM import  GenHMM, save_model, load_model
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import distributions

from datetime import datetime as dt

class TheDataset(Dataset):
    """Silly wrapper for DataLoader input."""
    def __init__(self, xtrain, lengths, device='cpu'):
        self.data = [torch.FloatTensor(x).to(device) for x in xtrain]
        self.lengths = lengths
        self.len=len(self.data)
        

    def __getitem__(self, index):
        mask = torch.cat( (torch.ones(self.lengths[index], dtype=torch.uint8), \
                           torch.zeros(self.data[index].shape[0] \
                                       - self.lengths[index], dtype=torch.uint8)) )
        return (self.data[index], mask)

    def __len__(self):
        return self.len


def pad_data(x,length):
    """ Add zeros at the end of all sequences in to get sequences of lengths `length`
    Input:  x : list, all of sequences of variable length to pad
            length : integer, common target length of sequences.
    output: list,  all input sequnces zero-padded.
    """

    d = x[0].shape[1]
    return [np.concatenate((xx,np.zeros((length - xx.shape[0] + 1,d)))) for xx in x]



if __name__ == "__main__":
    usage = "python bin/train_class.py data/train13.pkl models/epoch1_class1.mdlc"
    if len(sys.argv) != 3 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    # Parse
    train_inputfile = sys.argv[1]
    out_mdl = sys.argv[2]

    epoch_str, iclass_str = parse('epoch{}_class{}.mdlc',os.path.basename(out_mdl))
    train_class_inputfile = train_inputfile.replace(".pkl", "_{}.pkl".format(iclass_str))

    #  Load data
    xtrain = pkl.load(open(train_class_inputfile, "rb"))

    # Get the length of all the sequences
    l = [x.shape[0] for x in xtrain]

    #  Load or create model
    if epoch_str == '1':

        #  Create model, to
        options = dict(n_states=5, n_prob_components=3,
                       em_skip=30, device='cpu', lr=1e-4, log_dir="results")

        mdl = GenHMM(**options)

    else:
        # Load previous model

        mdl = load_model(out_mdl.replace("epoch" + epoch_str, "epoch" + str(int(epoch_str)-1)))

    
    mdl.device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        mdl.device = device
    
    print("Push model to {}...".format(mdl.device), file=sys.stderr)
    for s in range(mdl.n_states):
        for k in range(mdl.n_prob_components):
            mdl.networks[s,k] = mdl.networks[s,k].to(mdl.device)
            p = mdl.networks[s,k].prior
            mdl.networks[s,k].prior.loc = p.loc.to(mdl.device)#,p.covariance_matrix.to(mdl.device)).to(mdl.device)
            mdl.networks[s,k].prior.covariance_matrix = p.covariance_matrix.to(mdl.device)

    mdl.startprob_ = mdl.startprob_.to(mdl.device)
    mdl.transmat_ = mdl.transmat_.to(mdl.device)
    mdl.logPIk_s = mdl.pi.log().to(mdl.device)
    
    
    # zero pad data for batch training
    max_len_ = max([x.shape[0] for x in xtrain])
    xtrain_padded = pad_data(xtrain, max_len_)
    traindata = DataLoader(dataset=TheDataset(xtrain_padded, lengths=l, device=mdl.device), batch_size=256, shuffle=True)
    

    # niter counts the number of em steps before saving a model checkpoint
    niter = 1
    
    # em_skip determines the number of back-props before an EM step is performed
    mdl.em_skip = 5
    
    # TODO: pass lr as a param
    mdl.lr = 1e-3

    for iiter in range(niter):
        mdl.fit(traindata)
        
    save_model(mdl, fname=out_mdl)
    sys.exit(0)


