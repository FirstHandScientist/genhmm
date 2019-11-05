# file genHMM.py

import sys
import numpy as np
from gm_hmm.src.realnvp import RealNVP
import torch
from torch import nn, distributions
from gm_hmm.src._torch_hmmc import _compute_log_xi_sum, _forward, _backward
from gm_hmm.src.utils import step_learning_rate_decay, load_model, save_model, to_device, data_read_parse,pad_data,TheDataset, get_freer_gpu
from torch.utils.data import DataLoader


class GenHMMclassifier(nn.Module):
    def __init__(self, mdlc_files=None, **options):
        """Initialize a model on CPU. Make sure to push to GPU at runtime."""
        super(GenHMMclassifier, self).__init__()

        if mdlc_files == None:
            self.nclasses = options["nclasses"]
            self.hmms = [GenHMM(**options) for _ in range(self.nclasses)]
            self.pclass = torch.ones(len(self.hmms))

        else:
            self.mdlc_files = mdlc_files
            self.hmms = [load_model(fname) for fname in mdlc_files]
            self.pclass = torch.FloatTensor([h.number_training_data for h in self.hmms])
            self.pclass = (self.pclass / self.pclass.sum())

    # consider do linear training based on GenHMMs
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

    def fine_tune(self, use_gpu=False, Mul_gpu=False, batch_size=64):
        device = get_freer_gpu()

        print("device:", get_freer_gpu())
        self.hmms = [genhmm.train().pushto(get_freer_gpu()) for genhmm in self.hmms]
        self.pclass = self.pclass.reshape(-1,1).to(get_freer_gpu())
        data = [data_read_parse(genhmm.train_data_fname, dim_zero_padding=True) for genhmm in self.hmms]
        lengths = [[x.shape[0] for x in xtrain_class] for xtrain_class in data]
        max_len_ = max([max(l) for l in lengths])
        data = [pad_data(xtrain, max_len_) for xtrain in data]
        Y = np.concatenate([(int(g.iclass) - 1)*np.ones(len(classdata)) for g, classdata in zip(self.hmms, data)])
        Y = torch.ByteTensor(Y)
        train_data = DataLoader(dataset=TheDataset(sum(data, []),
                                                   ytrain=Y,
                                                   lengths=sum(lengths, []),
                                                   max_len_=max_len_,
                                                   device=get_freer_gpu()),
                                 batch_size=batch_size,
                                 shuffle=True)

        for i, genhmm in enumerate(self.hmms):
            self.hmms[i].optimizer = torch.optim.Adam(
                sum([[p for p in flow.parameters() if p.requires_grad == True] for flow in
                     self.hmms[i].networks.reshape(-1).tolist()], []), lr=self.hmms[i].lr)
            ada_lr = step_learning_rate_decay(init_lr=self.hmms[i].lr,
                                          global_step=self.hmms[i].global_step,
                                          minimum=1e-4,
                                          anneal_rate=0.98)

            for j, param_group in enumerate(self.hmms[i].optimizer.param_groups):
                self.hmms[i].optimizer.param_groups[j]['lr'] = ada_lr

            self.hmms[i].optimizer.load_state_dict(self.hmms[i].optimizer.state_dict())

        with torch.enable_grad():
            for b in train_data:
                for genhmm in self.hmms:
                    genhmm.optimizer.zero_grad()

                llh = torch.stack([genhmm.get_logprob(genhmm.networks, b[:-1]) for genhmm in self.hmms]).squeeze()
                log_pclass = self.pclass.log()
                denom = torch.logsumexp(llh + log_pclass, dim=0)
                print("b[-1]",b[-1].shape)
                y = torch.stack([~b[-1], b[-1]])
                num = llh[y] + log_pclass.repeat(1, y.shape[1])[y]
                
                loss = - (num - denom).sum()/float(batch_size)
                loss.backward()

                for genhmm in self.hmms:
                    genhmm.optimizer.step()

        for genhmm in self.hmms:
            genhmm._update_old_networks()
            genhmm.old_eval()
            genhmm.eval()
            genhmm.pushto("cpu")

        print("here")
        return self

    def save_members(self):
        for genhmm, out_mdl in zip(self.hmms, self.mdlc_files):
            save_model(genhmm, fname=out_mdl)
        return self

    def pushto(self, device):
        self.hmms = [h.pushto(device) for h in self.hmms]
        self.pclass = self.pclass.to(device)
        self.device = device
        return self

    def eval(self):
        for the_mdl in self.hmms:
            the_mdl.old_eval()
            the_mdl.eval()


class GenHMM(torch.nn.Module):
    def __init__(self, n_states=None, n_prob_components=None, device='cpu',\
                 dtype=torch.FloatTensor, \
                 EPS=1e-12, lr=None, em_skip=None,
                 net_H=28, net_D=14, net_nchain=10, mask_type="cross", p_drop=0.25,
                 startprob_type="first", transmat_type="random upper triangular"):
        super(GenHMM, self).__init__()

        self.n_states = n_states
        self.dtype = dtype
        self.n_prob_components = n_prob_components
 
        self.device = device
        self.dtype = dtype
        self.EPS = EPS
        self.lr = lr
        self.em_skip = em_skip
        
        # Initialize HMM parameters
        self.init_transmat(transmat_type)
        self.init_startprob(startprob_type)
        
        # Initialize generative model networks
        self.init_gen(H=net_H, D=net_D, nchain=net_nchain,
                      mask_type=mask_type, p_drop=p_drop)
        self._update_old_networks()
        self.old_eval()
        self.update_HMM = False

        # set the global_step
        self.global_step = 0

    def init_startprob(self, startprob_type="random"):
        """
        Initialize HMM initial coefficients.
        """
        if "random" in startprob_type:
            init = torch.abs(torch.randn(self.n_states))
            init /= init.sum()
            self.startprob_ = init

        elif "first" in startprob_type:
            init = torch.zeros(self.n_states)
            init[0] = 1
            self.startprob_ = init

        elif "uniform" in startprob_type:
            self.startprob_ = torch.ones(self.n_states)

        normalize(self.startprob_, axis=0)
        return self

    def init_transmat(self, transmat_type="random upper triangular"):
        """
        Initialize HMM transition matrix.
        """
        if "random" in transmat_type:
            self.transmat_ = (torch.randn(self.n_states, self.n_states)).abs()

        elif "uniform" in transmat_type:
            init = 1/self.n_states
            self.transmat_ = torch.ones(self.n_states, self.n_states) * init

        elif "triangular" in transmat_type:
            # use upper tra matrix
            self.transmat_ = (torch.randn(self.n_states, self.n_states)).abs()
            for i in range(self.n_states):
                for j in range(self.n_states):
                    if i > j:
                        self.transmat_[i, j] = 0

        elif "ergodic" in transmat_type:
            self.transmat_ = torch.ones(self.n_states, self.n_states) + torch.randn(self.n_states, self.n_states) * 0.01

        normalize(self.transmat_, axis=1)
        return self


    def init_gen(self, H, D, nchain, mask_type="cross", p_drop=0.25):
        """
        Initialize HMM probabilistic model.
        """
        d = D // 2

        nets = lambda: nn.Sequential(nn.Linear(d, H), nn.LeakyReLU(), 
                                     nn.Linear(H, H), nn.LeakyReLU(), 
                                     nn.Dropout(p_drop), nn.Linear(H, D))
        # Set mask
        if mask_type == "chunk":
            masks = torch.from_numpy(np.array([[0]*d + [1]*(D-d), [1]*d + [0]*(D-d)] * nchain).astype(np.bool))
        elif mask_type == "cross":
            masks = torch.from_numpy(np.array([[0, 1]*d, [1, 0]*d] * nchain).astype(np.bool))
        elif mask_type == "conv":
            # To do
            pass
        try:
            masks
        except NameError:
            print("masks are not defined")
            assert False

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
        self.optimizer = torch.optim.Adam(
            sum([[p for p in flow.parameters() if p.requires_grad == True] for flow in self.networks.reshape(-1).tolist()], []), lr=self.lr)
        return self

    def _update_old_networks(self):
        """load the parameters in self.networks (the one being optimized), into self.old_networks"""
        for i in range(self.n_states):
            for j in range(self.n_prob_components):
                self.old_networks[i, j].load_state_dict(self.networks[i, j].state_dict())
        return self

    def _affirm_networks_update(self):
        """affirm the parameters in self.networks are the same as self.old_networks"""
        
        for i in range(self.n_states):
            for j in range(self.n_prob_components):
                state_dict = self.networks[i,j].state_dict()
                for key, value in state_dict.items():
                    assert self.old_networks[i,j].state_dict()[key] == value
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
        self.pi = self.pi.to(device)
        self.logPIk_s = self.logPIk_s.to(device)
        
        self.device = device
        return self
    
    def old_eval(self):
        for s in range(self.n_states):
            for k in range(self.n_prob_components):
                # set old network mode as eval model
                self.old_networks[s,k].eval()
        return self
    

    def eval(self):
        for s in range(self.n_states):
            for k in range(self.n_prob_components):
                # set mode as eval model
                self.networks[s,k].eval()
        return self
    
    def train(self):
        for s in range(self.n_states):
            for k in range(self.n_prob_components):
                # set model as train mode
                self.networks[s,k].train()
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
        self.stats['start'] += posteriors[:,0].sum(dim=0)

        
        
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
        ### dong: original update here was not correct

        local_loglh_sk = loglh_sk
        max_loglh = torch.max(local_loglh_sk, dim=3, keepdim=True)[0]
        # should be careful to use the minus max trick here
        gamma_ = self.pi.reshape(1, 1, self.n_states, self.n_prob_components) * \
                 (local_loglh_sk - max_loglh).exp()
        gamma_ = gamma_ / (gamma_.sum(3, keepdim=True) +  self.EPS)
        # # set the elements corresponding to padded values to be zeros, this is done by zeroes in posteriors
        gamma = posteriors.unsqueeze(dim=3) * gamma_


        # In- line test for gamma computation, set the if condition to be true to compare gamm and statcs_prob_components
        if False:
            statcs_prob_components =torch.zeros(batch_size, n_samples, self.n_states, self.n_prob_components, device=self.device)
    #        print(max_loglh.shape)

            gamma_ = torch.zeros(batch_size, n_samples, self.n_states, self.n_prob_components, device=self.device)
            for b in range(batch_size):
                for n in range(n_samples):
                    for i in range(self.n_states):
                        for k in range(self.n_prob_components):
                            gamma_[b,n,i, k] = self.pi[i, k] * (local_loglh_sk[b, n, i, k]-local_loglh_sk[b,n,i,:].max()).exp()

                        gamma_[b,n,i,:] = gamma_[b, n, i, :] / (gamma_[b, n, i, :].sum() + self.EPS)

                        statcs_prob_components[b,n,i,:] = posteriors[b, n, i] * gamma_[b, n, i, :]

        self.stats["mixture"] += gamma.sum(1).sum(0)

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

    def get_logprob(self, networks, batch):
        x, x_mask = batch
        batch_size = x.shape[0]
        n_samples = x.shape[1]

        llh, _ = self._getllh(networks, batch)
        llh[~x_mask] = 0
        logprob, _ = self._do_forward_pass(llh, x_mask)
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
            loglh_sk = [networks[s, k].log_prob(x, x_mask)/x.size(2) for k in range(self.n_prob_components)]
            ll = torch.stack(loglh_sk).permute(1, 2, 0)
            local_loglh_sk[:, :, s, :] = ll
            llh[:, :, s] = torch.logsumexp((self.logPIk_s[s].reshape(1, 1, self.n_prob_components) + ll), dim=2)
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
        return loss, logprob.sum()


    def fit(self, traindata):
        """Performs one EM step and `em_skip` backprops before returning. The optimizer is re-initialized after each EM step.
            Follow the loss in stderr
            Input : traindata : torch.data.DataLoader object wrapping the batches.
            Output : None
        """
        # get the adaptive learning rate
        ada_lr = step_learning_rate_decay(init_lr=self.lr,
                                          global_step=self.global_step,
                                          minimum=1e-4,
                                          anneal_rate=0.98)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = ada_lr
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        # total number of sequences
        n_sequences = len(traindata.dataset)
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
                self.optimizer.zero_grad()            
                loss, logprob_ = self.forward(data, testing=False)
                loss.backward()
            
                self.optimizer.step()
                total_loss += loss.detach().data
                total_logprob += logprob_
            
            # consider put a stop criteria here to 
            
            print("epoch:{}\tclass:{}\tStep:{}\tb:{}\tLoss:{}\tNLL:{}".format(self.iepoch,
                                                                              self.iclass,i, b,
                                                                              total_loss/(b+1),
                                                                              -total_logprob/n_sequences), file=sys.stdout)
            
            
    
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
        tmp_pi = self.pi.clone()
        self.pi = self.stats["mixture"]
        normalize(self.pi, axis=1)
        # In case we get a line of zeros in the stats, skip the update brought by self.stats["mixture"]
        correct_idx = self.pi.sum(1).isclose(torch.ones(1, device=self.pi.device))
        self.pi[~ correct_idx] = tmp_pi[~ correct_idx]
        
        # any zero element in self.pi would cause -inf in self.logPIk_s. Fix: replace with self.EPS
        zero_idx = self.pi.isclose(torch.zeros(1, device=self.pi.device))
        self.pi[zero_idx] = self.EPS
        # normalize again
        normalize(self.pi, axis=1)
        # get log of pi
        self.logPIk_s = self.pi.log()


        # update output probabilistic model, networks here
        self._update_old_networks()
        self.old_eval()
        
        # store the latest NLL of the updated GenHMM model
        log_p_all = torch.cat(list(map(self.pred_score, traindata)))
        self.latestNLL = -log_p_all.sum()/n_sequences
        # store the average lop_p
        self.avrg_log_p = log_p_all.reshape(1, -1).squeeze().logsumexp(0)
        # stoe the max log_p
        self.max_log_p = log_p_all.max()

        print("epoch:{}\tclass:{}\tLatest NLL:\t{}".format(self.iepoch,self.iclass,self.latestNLL),file=sys.stdout)

        # Flag back to False
        self.update_HMM = False
        # set global_step
        self.global_step += 1


class wrapper(torch.nn.Module):
    def __init__(self, mdl):
        super(wrapper, self).__init__()
        self.userdata = mdl



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
