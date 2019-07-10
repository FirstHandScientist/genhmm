# file genHMM.py

import os
import sys
sys.path.append("..")
import numpy as np
from src.realnvp import RealNVP
import torch
from torch import nn, distributions
from src._torch_hmmc import _compute_log_xi_sum, _forward, _backward


class GenHMMclassifier(nn.Module):
    def __init__(self, mdlc_files=None, **options):
        """Initialize a model on CPU. Make sure to push to GPU at runtime."""
        super(GenHMMclassifier, self).__init__()

        if mdlc_files == None:
            self.nclasses = options["nclasses"]
            self.hmms = [GenHMM(**options) for _ in range(self.nclasses)]
            self.pclass = torch.ones(len(self.hmms))

        else:
            self.hmms = [load_model(fname) for fname in mdlc_files]
            self.pclass = torch.FloatTensor([h.number_training_data for h in self.hmms])
            self.pclass = (self.pclass / self.pclass.sum())
        

    ### consider do linear training based on GenHMMs
    def forward(self, x, weigthed=False):
        """compute likelihood of data under each GenHMM
        INPUT:
        x: The torch batch data
           or x should be tuple: (batch size * n_samples (padded length) * n_features, 
                                  tensor mask of the batch data)
        
        OUTPUT: tensor of likelihood, shape: data_size * ncl
        """

        if weigthed:
            batch_llh = [classHMM.pred_score(x) / classHMM.latestNLL for classHMM in self.hmms]
        else:
            batch_llh = [classHMM.pred_score(x) for classHMM in self.hmms]

        return torch.stack(batch_llh)


    def pushto(self, device):
        self.hmms = [h.pushto(device) for h in self.hmms]
        self.pclass = self.pclass.to(device)
        self.device = device
        return self


class GenHMM(torch.nn.Module):
    def __init__(self, n_states=None, n_prob_components=None, device='cpu',\
                 dtype=torch.FloatTensor, \
                 EPS=1e-12, lr=None, em_skip=None,
                 net_H=28, net_D=14, net_nchain=10):
        super(GenHMM, self).__init__()

        self.n_states = n_states
        self.dtype = dtype
        self.n_prob_components = n_prob_components
 
        self.device=device
        self.dtype=dtype
        self.EPS = EPS
        self.lr = lr
        self.em_skip = em_skip
        
        # Initialize HMM parameters
        self.init_transmat()
        self.init_startprob()
        
        # Initialize generative model networks
        self.init_gen(H=net_H, D=net_D, nchain=net_nchain)
        self._update_old_networks()
        self.update_HMM = False

        
    def init_startprob(self, random=True):
        """
        Initialize HMM initial coefficients.
        """
        if random:
            init = torch.abs(torch.randn(self.n_states))
            init /= init.sum()
            self.startprob_ = init
        else:
            init = 1. / self.n_states
            self.startprob_ = torch.ones(self.n_states) * init
            
        return self

    def init_transmat(self, random=True):
        """
        Initialize HMM transition matrix.
        """
        if random:
            self.transmat_ = torch.randn(self.n_states, self.n_states).abs()
            normalize(self.transmat_, axis=1)
        else:
            init = 1/self.n_states
            self.transmat_ = torch.ones(self.n_states, self.n_states) * init
        return self

    def init_gen(self, H, D, nchain):

        """
        Initialize HMM probabilistic model.
        """
        d = D // 2

        nets = lambda: nn.Sequential(nn.Linear(d, H), nn.LeakyReLU(), nn.Linear(H, H), nn.LeakyReLU(), nn.Linear(H, D))
        
        masks = torch.from_numpy(np.array([[0]*d + [1]*(D-d), [1]*d + [0]*(D-d)] * nchain).astype(np.uint8))
        ### torch MultivariateNormal logprob gets error when input is cuda tensor
        ### thus changing it to implementation
        prior = distributions.MultivariateNormal(torch.zeros(D).to(self.device), torch.eye(D).to(self.device))
        # prior = lambda x: GaussianDiag.logp(torch.zeros(D), torch.zeros(D), x)
        # self.flow = RealNVP(nets, nett, masks, prior)


        #  Init mixture
        self.pi = self.dtype(np.random.rand(self.n_states, self.n_prob_components))
        normalize(self.pi, axis=1)

        self.logPIk_s = self.pi.log()


        # Init networks
        self.networks = [RealNVP(nets, masks, prior) for _ in range(self.n_prob_components*self.n_states)]

        # Reshape in a n_states x n_prob_components array
        self.networks = np.array(self.networks).reshape(self.n_states, self.n_prob_components)
        
        # initial an old networks for posterior computations with the same sturcture
        self.old_networks = [RealNVP(nets, masks, prior) for _ in range(self.n_prob_components*self.n_states)]
        self.old_networks = np.array(self.old_networks).reshape(self.n_states, self.n_prob_components)
        return self
    
    def _update_old_networks(self):
        """load the parameters in self.networks (the one being optimized), into self.old_networks"""
        for i in range(self.n_states):
            for j in range(self.n_prob_components):
                self.old_networks[i,j].load_state_dict( self.networks[i,j].state_dict() )
        return self
    
    def pushto(self, device):
        for s in range(self.n_states):
            for k in range(self.n_prob_components):
                # push new networks to device
                self.networks[s,k].to(device)
                p = self.networks[s,k].prior
                self.networks[s,k].prior = type(p)(p.loc.to(device),
                                                  p.covariance_matrix.to(device))
                
                # push the old networks to device
                self.old_networks[s,k].to(device)
                p = self.old_networks[s,k].prior
                self.old_networks[s,k].prior = type(p)(p.loc.to(device),
                                                  p.covariance_matrix.to(device))
                
        self.startprob_ = self.startprob_.to(device)
        self.transmat_ = self.transmat_.to(device)
        self.logPIk_s = self.logPIk_s.to(device)

        self.device = device
        return self

    def _initialize_sufficient_statistics(self):
        """Initializes sufficient statistics required for M-step.

        The method is *pure*, meaning that it doesn't change the state of
        the instance.  For extensibility computed statistics are stored
        in a dictionary.

        Returns
        -------
        nobs : int
            Number of samples in the data.

        start : array, shape (n_components, )
            An array where the i-th element corresponds to the posterior
            probability of the first sample being generated by the i-th
            state.

        trans : array, shape (n_components, n_components)
            An array where the (i, j)-th element corresponds to the
            posterior probability of transitioning between the i-th to j-th
            states.
        """

        stats = {'nobs': 0,
                 'nframes':0,
                 'start': torch.zeros(self.n_states).to(self.device),
                 'trans': torch.zeros(self.n_states, self.n_states).to(self.device),
                 'mixture': torch.zeros(self.n_states, self.n_prob_components).to(self.device),
                 'loss': torch.FloatTensor([0]).to(self.device)
                 }

        self.stats = stats

   
    def _accumulate_sufficient_statistics(self, framelogprob, mask,
                                          posteriors, logprob, fwdlattice,
                                          bwdlattice, loglh_sk):
        """Updates sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~base._BaseHMM._initialize_sufficient_statistics`.

        framelogprob : array, shape (batch_size, n_samples, n_components)
            Log-probabilities of each sample under each of the model states.

        posteriors : array, shape (batch_size, n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.

        fwdlattice, bwdlattice : array, shape (batch_size, n_samples, n_components)
            Log-forward and log-backward probabilities.

        loglh_sk : array, shape (batch_size, n_samples, n_components, n_prob_components)
            Log-probabilities of each batch sample under each components of each states.
        """


        batch_size, n_samples, n_components = framelogprob.shape

        self.stats['nframes'] += mask.sum()
        self.stats['nobs'] += batch_size
        self.stats['start'] += posteriors[:,0].sum(0)

        
        
        log_xi_sum = _compute_log_xi_sum(n_samples, n_components, fwdlattice,
                                         torch.log(self.transmat_ + self.EPS),
                                         bwdlattice, framelogprob,
                                         torch.ones(batch_size,
                                                    n_components,
                                                    n_components,
                                                    device=self.device) * float('-inf'),
                                         logprob, mask)
            # _log_xi_sum = _compute_log_xi_sum(n_samples, n_components,\
            #                                    self.dtype(fwdlattice).to(self.device),\
            #                                    self.dtype(log_mask_zero(self.transmat_)).to(self.device),\
            #                                    self.dtype(bwdlattice).to(self.device), \
            #                                    self.dtype(framelogprob).to(self.device),\
            #                                    self.dtype(np.full((n_components, n_components), -np.inf)).to(self.device))

            
        self.stats['trans'] += torch.exp(log_xi_sum).sum(0)

 #       print(loglh_sk.shape, self.n_states, self.n_prob_components)
        # max_loglh = torch.max(torch.max(loglh_sk, dim=1)[0],dim=1)[0]
        ### dong: have not verify this computation
        local_loglh_sk = loglh_sk.reshape(batch_size, self.n_states, self.n_prob_components, n_samples)
        max_loglh = torch.max(torch.max(local_loglh_sk, dim=3)[0], dim=2)[0]
#        print(max_loglh.shape)

        gamma_ = torch.zeros(batch_size, self.n_states, self.n_prob_components, n_samples, device=self.device)

        for i in range(self.n_states):
            for t in range(n_samples):
                for m in range(self.n_prob_components):
                    gamma_[:,i, m, t] = self.pi[i, m] * (local_loglh_sk - max_loglh[:,i].reshape(-1,1,1,1))[:, i, m, t].exp()
                
 #               print(gamma_[:, i, :, t].shape,  gamma_[:,i, :, t].sum(1).shape) 
                gamma_[:, i, :, t] /= (gamma_[:,i, :, t].sum(1).reshape(-1, 1) + 1e-6)   #.reshape(self.n_states,1,n_samples)
#                print(gamma_[:, i, :, t].shape, posteriors[:, t, i].shape)
 
                gamma_[:, i, :, t] *= posteriors[:, t, i].reshape(-1,1)

        self.stats["mixture"] += gamma_.sum(3).sum(0)

        return



    def _do_forward_pass(self, framelogprob, mask):
        batch_size, n_samples, n_components = framelogprob.shape
        # in case log computation encounter log(0), do log(x + self.EPS)
        
        
        ### To Do: matain hmm parameters as torch tensors
        log_startprob = torch.log(self.startprob_ + self.EPS)
        log_transmat = torch.log(self.transmat_ + self.EPS)

        return _forward(n_samples, n_components, log_startprob, \
                        log_transmat, framelogprob, mask) 

    def _do_backward_pass(self, framelogprob, mask):
        batch_size, n_samples, n_components = framelogprob.shape
        
        ### To Do: matain hmm parameters as torch tensors
        log_startprob = torch.log(self.startprob_ + self.EPS)
        log_transmat = torch.log(self.transmat_ + self.EPS)
        return _backward(n_samples, n_components, log_startprob,\
                         log_transmat, framelogprob, mask)
        
    def _compute_posteriors(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        # Normalizes the input array so that the exponent of the sum is 1
        lse_gamma = torch.logsumexp(log_gamma, dim=2)
        
        log_gamma -= lse_gamma[:, :, None]
        
        return torch.exp(log_gamma)

    def pred_score(self, X):
        """ Update the base score method, such that the scores of sequences are returned
        score: the log probability under the model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        Returns
        -------
        logprob : list of floats, [logprob1, logprob2, ... ]
            Log likelihood of ``X``.
        """
        # now mask is used, need to pass mask as well
        # will consider to do batch as well in testig
        # mask = torch.ones(1, lengths[0], dtype=torch.uint8)
        # X = self.dtype(X[None,:]).to(self.device)
        logprob = self.forward(X, testing=True)
        return logprob
    
    def _getllh(self, networks, batch):
        """Computing the llh and loglh_sk"""
        x, x_mask = batch
        batch_size = x.shape[0]
        n_samples = x.shape[1]

        llh = torch.zeros(batch_size, n_samples, self.n_states).to(self.device)

        local_loglh_sk = torch.zeros((batch_size, n_samples, self.n_states, self.n_prob_components)).to(self.device)

        # sequences_true_len = x_mask.type_as(llh).sum(1).reshape(-1, 1, 1)

        # TODO: some parallelization here...
        for s in range(self.n_states):
            loglh_sk = [0 for _ in range(self.n_prob_components)]

            for k in range(self.n_prob_components):
                loglh_sk[k] = networks[s, k].log_prob(x, x_mask).reshape(batch_size, 1, -1)/x.size(2)
                #assert((loglh_sk[k] <= 0).all())

            # loglh_sk = [llh_sk / sequences_true_len for llh_sk in loglh_sk ]

            ll = torch.cat(loglh_sk, dim=1)
            local_loglh_sk[:, :, s, :] = ll.transpose(1, 2)
            llh[:, :, s] = (self.logPIk_s[s].reshape(1, self.n_prob_components, 1) + ll).detach().sum(1)
        return llh, local_loglh_sk



    def forward(self, batch, testing=False):
        """PYTORCH FORWARD, NOT HMM forward algorithm. This function is called for each batch.
        Input: batch of sequences, array size, (batch_size, n_samples, n_dimensions)
        Output: Loss, scaler
        """

        if self.update_HMM and not testing:
            self._initialize_sufficient_statistics()
        
        x, x_mask = batch
        batch_size = x.shape[0]
        n_samples = x.shape[1]

        # get the log-likelihood for posterior computation
        with torch.no_grad():
            # Two posteriors to be computed here:
            # 1. the hidden state posterior, post
            old_llh, old_loglh_sk = self._getllh(self.old_networks, batch)
            old_llh[~x_mask] = 0
            old_logprob, old_fwdlattice = self._do_forward_pass(old_llh, x_mask)
            # assert ((old_logprob <= 0).all())

            if testing:
                # each EM step sync old_networks and networks, so it is ok to test on old_networks
                return old_logprob

            old_bwdlattice = self._do_backward_pass(old_llh, x_mask)
            posteriors = self._compute_posteriors(old_fwdlattice, old_bwdlattice)

            posteriors[~x_mask] = 0
            post = posteriors

            # 2. the probability model components posterior, k condition on hidden state, observation and hmm model
            # Compute log-p(chi | s, X) = log-P(X|s,chi) + log-P(chi|s) - log\sum_{chi} exp ( log-P(X|s,chi) + log-P(chi|s) )
        
            log_num = old_loglh_sk.detach() + self.logPIk_s.reshape(1, self.n_states, self.n_prob_components)
            #log_num = brackets.detach()
            log_denom = torch.logsumexp(log_num, dim=3)
            
            logpk_sX = log_num - log_denom.reshape(batch_size, n_samples, self.n_states, 1)
            ## To Do: normalize logpk_sX before set un-masked values
            logpk_sX[~x_mask] = 0
        
        # hmm parameters should be updated based on old model
        if self.update_HMM and not testing:
            self._accumulate_sufficient_statistics(old_llh, x_mask,
                                                   posteriors, old_logprob,
                                                   old_fwdlattice, old_bwdlattice, old_loglh_sk)

        # Get the log-likelihood to format cost such self.networks such it can be optimized
        llh, self.loglh_sk = self._getllh(self.networks, batch)

        # compute sequence log-likelihood in self.networks, just to monitor the self.networks performance
        with torch.no_grad():
            llh[~x_mask] = 0
            logprob, _ = self._do_forward_pass(llh, x_mask)
        # assert((logprob <= 0).all())
        # Brackets = log-P(X | chi, S) + log-P(chi | s)
        brackets = torch.zeros_like(self.loglh_sk)
        ## Todo: implement update pi_s_k ?
        brackets[x_mask] = self.loglh_sk[x_mask] + self.logPIk_s.reshape(1, self.n_states, self.n_prob_components)
        
        #  The .sum(3) call sums on the components and .sum(2).sum(1) sums on all states and samples
        # loss = -(post * (torch.exp(logpk_sX) * brackets).sum(3)).sum(2).sum(1).sum()/float(x_mask.sum())
        loss = -(post[x_mask] * (torch.exp(logpk_sX) * brackets)[x_mask].sum(2)).sum()/float(batch_size)
        return loss, logprob.mean()


    def fit(self, traindata):
        """Performs one EM step and `em_skip` backprops before returning. The optimizer is re-initialized after each EM step.
            Follow the loss in stderr
            Input : traindata : torch.data.DataLoader object wrapping the batches.
            Output : None
        """

        optimizer = torch.optim.Adam(
            sum([[p for p in flow.parameters() if p.requires_grad == True] for flow in self.networks.reshape(-1).tolist()], []), lr=self.lr)
        

        for i in range(self.em_skip):
            # if i is the index of last loop, set update_HMM as true

            if i == self.em_skip-1:
                self.update_HMM = True
            else:
                self.update_HMM = False


           
            total_loss = 0
            total_logprob = 0
            for b, data in enumerate(traindata):
                # start = dt.now()
                optimizer.zero_grad()            
                loss, logprob_ = self.forward(data, testing=False)
                loss.backward()
            
                optimizer.step()
                total_loss += loss.detach().data
                total_logprob += logprob_
            
            # consider put a stop criteria here to 
            print("epoch:{}\tclass:{}\tStep:{}\tb:{}\tLoss:{}\tNLL:{}".format(self.iepoch,self.iclass,i, b,
                                               total_loss/(b+1),
                                               -total_logprob/(b+1)),
                  file=sys.stdout)
            
            
    
        # Perform EM step
        # Update initial proba
        # startprob_ = self.startprob_prior - 1.0 + self.stats['start']
        startprob_ = self.stats['start']
        self.startprob_ = torch.where(self.startprob_ == 0.0,
                                   self.startprob_, startprob_)
        normalize(self.startprob_, axis=0)
        
        # Update transition
        # transmat_ = self.transmat_prior - 1.0 + self.stats['trans']
        transmat_ = self.stats['trans']
        self.transmat_ = torch.where(self.transmat_ == 0.0,
                                  self.transmat_, transmat_)
        normalize(self.transmat_, axis=1)
        
        # Update prior
        self.pi = self.stats["mixture"]

        # In case we get a line of zeros in the stats
        #self.pi[self.pi.sum(1) == 0, :] = np.ones(self.n_prob_components) / self.n_prob_components
        normalize(self.pi, axis=1)

        self.logPIk_s = self.pi.log()


        # update output probabilistic model, networks here
        self._update_old_networks()
        
        # store the latest NLL of the updated GenHMM model
        self.latestNLL = -torch.cat(list(map(self.pred_score, traindata))).mean()

        print("epoch:{}\tclass:{}\tLatest NLL:\t{}".format(self.iepoch,self.iclass,self.latestNLL),file=sys.stdout)

        # Flag back to False
        self.update_HMM = False

class wrapper(torch.nn.Module):
    def __init__(self, mdl):
        super(wrapper, self).__init__()
        self.userdata = mdl


def save_model(mdl, fname=None):
    torch.save(wrapper(mdl), fname)
    return 0


def load_model(fname):
    """Loads a model on CPU by default."""
    savable = torch.load(fname, map_location='cpu')
    return savable.userdata


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


