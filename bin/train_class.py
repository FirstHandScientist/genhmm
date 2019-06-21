import os
import sys
sys.path.append("..")

from parse import parse
import pickle as pkl
from src.genHMM import GenHMM, save_model, load_model
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import distributions

from datetime import datetime as dt

class TheDataset(Dataset):
    def __init__(self, xtrain):
        self.data = [torch.FloatTensor(x).to(device) for x in xtrain]
        self.len=len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def pad_data(x,length):
    d = x[0].shape[1]
    return [np.concatenate((xx,np.zeros((length - xx.shape[0] + 1,d)))) for xx in x]
 
def normalize(a, axis=None):
    """Normalizes the input array so that it sums to 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis, keepdim=True)
    a_sum[a_sum==0] = 1
    a /= a_sum



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
    #xtrain = xtrain#[:100]

    # Reshape data
    l = [x.shape[0] for x in xtrain]

    X = np.concatenate(xtrain)

    #  Load or create model
    if epoch_str == '1':
        #  Create model
        options = dict(n_components=5, n_prob_components=3,
                       n_iter=3000,
                       em_skip=30, tol=0)

        mdl = GenHMM(**options)
        

    else:
        # Load previous model
        mdl = load_model(out_mdl.replace("epoch" + epoch_str, "epoch" + str(int(epoch_str)-1)))


    if torch.cuda.is_available():
        device = torch.device('cuda')
        X = torch.DoubleTensor(X).to(device)
        mdl.push2gpu(device)
        mdl.device = device
    
    print("Start")
    
    mdl.logPIk_s = mdl.var_nograd(mdl.pi).log().to(mdl.device)
    curr_logprob = torch.FloatTensor([0]).to(mdl.device)
    stats = mdl._initialize_sufficient_statistics()
    start_ = dt.now()
    
    # Push all models to GPU
    for s in range(mdl.n_states):
        for k in range(mdl.n_prob_components):
            mdl.networks[s,k] = mdl.networks[s,k].to(mdl.device)
            p = mdl.networks[s,k].prior
            mdl.networks[s,k].prior.loc = p.loc.to(mdl.device)#,p.covariance_matrix.to(mdl.device)).to(mdl.device)
            mdl.networks[s,k].prior.covariance_matrix = p.covariance_matrix.to(mdl.device)


    class getllh(torch.nn.Module):
        def __init__(self, networks, n_states=None, n_prob_components=None, device='cpu', mdl=None):
            super(getllh, self).__init__()
            self.networks = networks
            self.n_states = n_states
            self.n_prob_components = n_prob_components
            self.device=device
            self.mdl=mdl

        def forward(self, x):
        # PYTORCH FORWARD, NOT HMM forward algorithm
            if self.mdl.update_HMM:
                self.mdl._initialize_sufficient_statistics()

            batch_size = x.shape[0]
            n_samples = x.shape[1]

            llh = torch.zeros(batch_size, n_samples, mdl.n_states).to(mdl.device)
            self.loglh_sk = torch.zeros((batch_size, self.n_states, self.n_prob_components, n_samples)).to(self.device)

            for s in range(mdl.n_states):
                loglh_sk = [mdl.networks[s, k].log_prob(x).reshape(batch_size, 1, -1)/x.numel() for k in range(self.n_prob_components)]
                ll = torch.cat(loglh_sk, dim=1)
                self.loglh_sk[:,s,:,:] = ll
                
                llh[:,:,s] = (self.mdl.logPIk_s[s].reshape(1,mdl.n_prob_components, 1) + ll).detach().sum(1)
           
            
            logprob, fwdlattice = self.mdl._do_forward_pass(llh)
            bwdlattice = self.mdl._do_backward_pass(llh)
            posteriors = self.mdl._compute_posteriors(fwdlattice, bwdlattice)
            
            if self.mdl.update_HMM:
                self.mdl._accumulate_sufficient_statistics(x, llh, posteriors, fwdlattice, bwdlattice, self.loglh_sk)


            # Compute loss associated with sequence
            logPIk_s_ext = self.mdl.logPIk_s.reshape(1, mdl.n_states, mdl.n_prob_components, 1)
            
            # Brackets = log-P(X | chi, S) + log-P(chi | s)
            brackets = self.loglh_sk + logPIk_s_ext

            # Compute log-p(chi | s, X) = log-P(X|s,chi) + log-P(chi|s) - log\sum_{chi} exp ( log-P(X|s,chi) + log-P(chi|s) )
            log_num = self.loglh_sk.detach() + logPIk_s_ext
            log_denom = torch.logsumexp(self.loglh_sk.detach() + logPIk_s_ext, dim=2)

            logpk_sX = log_num - log_denom.reshape(batch_size, self.n_states, 1, n_samples)

            ##### does the original swapaxes meas transpose????? please confirm
            #post = mdl.var_nograd(posteriors.swapaxes(0, 1))  # Transform into n_states x n_samples
            
            post = posteriors.transpose(1,2)
            # print(post.shape, (torch.exp(logpk_sX) * brackets).sum(2).shape)
            #  The .sum(2) call sums on the components and .sum(1).sum(1) sums on all states and samples
            loss = -(post * (torch.exp(logpk_sX) * brackets).sum(2)).sum(1).sum(1)/(n_samples*batch_size)
            return loss.sum()

    seq_llh_computer = getllh(networks=mdl.networks,n_states=mdl.n_states,n_prob_components=mdl.n_prob_components, device=mdl.device, mdl=mdl).to(mdl.device)
    
    
   
    
    lr=1e-4
   
    model = seq_llh_computer
    optimizer = torch.optim.Adam(
            sum([[p for p in flow.parameters() if p.requires_grad == True] for flow in model.networks.reshape(-1).tolist()], []), lr=lr)
    
    # Prepare data
    max_len_ = max([x.shape[0] for x in xtrain])
    xtrain_padded = pad_data(xtrain, max_len_)
    traindata = DataLoader(dataset=TheDataset(xtrain_padded), batch_size=64, shuffle=True)
    niter = 300
    em_skip = 20
    
    for iiter in range(niter):
        model.mdl.update_HMM = False

        for i in range(em_skip):
            if i == em_skip - 1:
               model.mdl.update_HMM = True

            optimizer.zero_grad()
            for b, data in enumerate(traindata):
                start = dt.now()
                loss = model(data)
                loss.backward()
            print("i:{}\tb:{}\tLoss:{}".format(i, b, loss.data))
            optimizer.step()
    
        # Perform EM step
        # Update initial proba
        startprob_ = model.mdl.startprob_prior - 1.0 + model.mdl.stats['start']
        model.mdl.startprob_ = torch.where(model.mdl.startprob_ == 0.0,
                                   model.mdl.startprob_, startprob_)
        normalize(model.mdl.startprob_, axis=0)
        
        # Update transition
        transmat_ = model.mdl.transmat_prior - 1.0 + model.mdl.stats['trans']
        model.mdl.transmat_ = torch.where(model.mdl.transmat_ == 0.0,
                                  model.mdl.transmat_, transmat_)
        normalize(model.mdl.transmat_, axis=1)
        
        # Update prior
        model.mdl.pi = model.mdl.stats["mixture"]

        # In case we get a line of zeros in the stats
        #self.pi[self.pi.sum(1) == 0, :] = np.ones(self.n_prob_components) / self.n_prob_components
        normalize(model.mdl.pi, axis=1)

        


    sys.exit(0)
    mdl.fit(X, lengths=l)
    save_model(mdl, fname=out_mdl)
    sys.exit(0)


