import os
import sys
sys.path.append("..")

from parse import parse
import pickle as pkl
from src.genHMM import GenHMM, save_model, load_model
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime as dt

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
    
    X_ = [torch.FloatTensor(x).to(device) for x in xtrain]
    print(X_[0])
    nseq = len(X_)
    print("Start")
    
    def criterion(llh):
        logprob, fwdlattice = mdl._do_forward_pass(llh)
        curr_logprob += logprob
        bwdlattice = mdl._do_backward_pass(llh)
        posteriors = mdl._compute_posteriors(fwdlattice, bwdlattice)

        # Compute loss associated with sequence
        logPIk_s_ext = mdl.var_nograd(mdl.logPIk_s.reshape(mdl.n_states, mdl.n_prob_components, 1))

        # Brackets = log-P(X | chi, S) + log-P(chi | s)
        brackets = mdl.loglh_sk + logPIk_s_ext

        # Compute log-p(chi | s, X) = log-P(X|s,chi) + log-P(chi|s) - log\sum_{chi} exp ( log-P(X|s,chi) + log-P(chi|s) )
        log_num = mdl.loglh_sk.detach() + logPIk_s_ext
        log_denom = mdl.var_nograd(torch.logsumexp(mdl.loglh_sk.detach() + logPIk_s_ext, dim=1))

        logpk_sX = log_num - log_denom.reshape(mdl.n_states, 1, n_samples)

        ##### does the original swapaxes meas transpose????? please confirm
        #post = mdl.var_nograd(posteriors.swapaxes(0, 1))  # Transform into n_states x n_samples
        post = posteriors.transpose(0,1)

        #  The .sum(1) call sums on the components and .sum() sums on all states and samples
        loss = -(post * (torch.exp(logpk_sX) * brackets).sum(1)).sum()/(n_samples)

        return loss
    
    
    mdl.logPIk_s = mdl.var_nograd(mdl.pi).log().to(mdl.device)
    curr_logprob = torch.FloatTensor([0]).to(mdl.device)
    stats = mdl._initialize_sufficient_statistics()
    start_ = dt.now()
    
    # Push all models to GPU
    for s in range(mdl.n_states):
        for k in range(mdl.n_prob_components):
            mdl.networks[s,k] = mdl.networks[s,k].to(mdl.device)

            print(mdl.networks[s,k].prior)

    for i,x in enumerate(X_[:200]):
        start = dt.now()
        
        # Compute llh of sequence
        n_samples = x.shape[0]
        llh = torch.zeros(n_samples, mdl.n_states).to(mdl.device)
        mdl.loglh_sk = mdl.var_nograd(np.zeros((mdl.n_states, mdl.n_prob_components, n_samples))).to(mdl.device)

        for s in range(mdl.n_states):
            loglh_sk = [mdl.networks[s, k].log_prob(x).reshape(1, -1)/x.numel() for k in range(mdl.pi[s].shape[0])]
            ll = torch.cat(loglh_sk, dim=0)
            mdl.loglh_sk[s] = ll
            llh[:,s] = mdl.var_nograd(mdl.logPIk_s[s].reshape(mdl.n_prob_components, 1) + ll).detach().sum(0)
        
       # if i % 50 == 0: 
        
        print("get_llh:{}/{}\t{}".format(i,nseq,(dt.now() - start).total_seconds()))
        
        start = dt.now()

        print("get_loss:{}/{}\t{}".format(i,nseq,(dt.now() - start).total_seconds()))
        loss = criterion(llh)

        loss.backward()
        stats['loss'] += loss.detach()

    ending = dt.now()
    print("elapsed:",(ending-start).total_seconds(), l) 
    
    sys.exit(0)
    mdl.fit(X, lengths=l)
    save_model(mdl, fname=out_mdl)
    sys.exit(0)


