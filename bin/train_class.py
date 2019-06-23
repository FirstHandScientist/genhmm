import os
import sys
sys.path.append("..")

from parse import parse
import pickle as pkl
from src.genHMM import GenHMM
from src.utils import save_model, load_model
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import distributions

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from src.utils import pad_data, normalize
from src.modules import TheDataset, getllh



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



    seq_llh_computer = getllh(networks=mdl.networks,n_states=mdl.n_states,n_prob_components=mdl.n_prob_components, device=mdl.device, mdl=mdl).to(mdl.device)
    
    
   
    
    lr=1e-4
   
    model = seq_llh_computer
    optimizer = torch.optim.Adam(
            sum([[p for p in flow.parameters() if p.requires_grad == True] for flow in model.networks.reshape(-1).tolist()], []), lr=lr)
    
    # Prepare data
    max_len_ = max([x.shape[0] for x in xtrain])
    xtrain_padded = pad_data(xtrain, max_len_)
    traindata = DataLoader(dataset=TheDataset(xtrain_padded,device), batch_size=32, shuffle=True)
    niter = 1
    em_skip = 50
    
    # initial a writer for loss and likelihood value record
    # log_dir = os.path.join(os.path.dirname(out_mdl), "class"+iclass_str)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # tensor_writer = SummaryWriter(log_dir=log_dir)
    
    for iiter in range(niter):
        model.mdl.update_HMM = False

        for i in range(em_skip):
            if i == em_skip - 1:
               model.mdl.update_HMM = True

            optimizer.zero_grad()
            llh = 0
            total_loss = 0
            for b, data in enumerate(traindata):
                start = dt.now()
                loss, batch_llh = model(data)
                loss.backward()
                with torch.no_grad():
                    llh += batch_llh
                    total_loss += loss
                
            print("[CLASS{}, EMstep:{}]: i:{}\tb:{}\tLoss:{}\t NLL:{}".format(iclass_str, epoch_str,i, b, total_loss.data, llh.data))
            # tensor_writer.add_scalar("NLL/epoch", -llh, iiter)
            # tensor_writer.add_scalar("Loss/epoch", loss, iiter)
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

        


    # sys.exit(0)
    # mdl.fit(X, lengths=l)
    save_model(mdl, fname=out_mdl)
    sys.exit(0)


