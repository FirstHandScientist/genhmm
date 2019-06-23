# file to put class modules 
import sys
sys.path.append("..")
import torch
from src.genHMM import GenHMM
from src.utils import load_model, save_model
from torch import nn
from torch.utils.data import Dataset, DataLoader


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
        return loss.sum(), llh.sum()


class GenHMMclassifier(nn.Module):
    def __init__(self, mdlc_files=None, **options):
        super(GenHMMclassifier, self).__init__()

        if mdlc_files == None:
            self.nclasses = options["nclasses"]
            self.hmms = [GenHMM(**options) for _ in range(self.nclasses)]
        else:
            self.hmms = [load_model(fname) for fname in mdlc_files]

        
    def forward(self, x, lengths=None):
        if lengths is None:
            # Assume we passed only one sequence.
            l = [x.shape[0]]

        else:
            l = lengths

        return [classHMM.pred_score(x, lengths=l)[0] for classHMM in self.hmms]

class TheDataset(Dataset):
    def __init__(self, xtrain,device):
        self.data = [torch.FloatTensor(x).to(device) for x in xtrain]
        self.len=len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class wrapper(torch.nn.Module):
    def __init__(self, mdl):
        super(wrapper, self).__init__()
        self.userdata = mdl
