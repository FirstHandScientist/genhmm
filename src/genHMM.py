# file genHMM.py

import numpy as np
from scipy.special import logsumexp as lsexp

from hmmlearn.utils import normalize, log_mask_zero
from src.glow.models import Glow, FlowNet
from src.glow.config import JsonConfig
from src.realnvp import RealNVP

from hmmlearn.base import _BaseHMM
from hmmlearn.base import _hmmc
from hmmlearn.hmm import GMMHMM
from functools import partial

import torch
from torch import nn, distributions
from torch.autograd import Variable


class GenHMMclassifier(nn.Module):
    def __init__(self, options, inp_dim):
        super(GenHMMclassifier, self).__init__()
        self.nclasses = options["nclasses"]
        self.hmms = [GenHMM(**options) for _ in range(self.nclasses)]

    def forward(self, x):
        return [classHMM.llh(x) for classHMM in self.hmms]


class GenHMM(_BaseHMM):
    def __init__(self, n_components=None, n_prob_components=None,
            algorithm="viterbi", random_state=None, n_iter=100, em_skip=10, tol=1e-2, verbose=False,
                 params="stmg", init_params="stmg", dtype=torch.FloatTensor):


        _BaseHMM.__init__(self, n_components, algorithm=algorithm, random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose, init_params=init_params)

        self.n_components = n_components
        # Handy renaming
        self.n_states = self.n_components
        self.dtype = dtype
        self.n_prob_components = n_prob_components
        self.params = params
        self.init_params = init_params
        self.em_skip = em_skip
        self.em_skip_cond = lambda: self.monitor_.iter % self.em_skip != 0 or self.monitor_.iter == 0 # or self.monitor_.iter == 0:

        self.model_params = ["transmat_", "startprob_", "networks", "pi"]

        self.init_transmat()
        self.init_startprob()

        self.init_gen()
        #self.init_future()


    def init_startprob(self):
        """
        Initialize HMM initial coefficients.
        """
        init = 1. / self.n_states
        self.startprob_ = np.full(self.n_states, init)
        return self

    def init_transmat(self):
        """
        Initialize HMM transition matrix.
        """
        init = 1/self.n_components
        self.transmat_ = np.full((self.n_states, self.n_states),
                                 init)
        return self

    def init_gen(self):
        """
        Initialize HMM probabilistic model.
        """
        H = 28
        D = 14
        nchain = 3
        d = D // 2

        nets = lambda: nn.Sequential(nn.Linear(D, H), nn.LeakyReLU(), nn.Linear(H, H), nn.LeakyReLU(), nn.Linear(H, D),
                                     nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(D, H), nn.LeakyReLU(), nn.Linear(H, H), nn.LeakyReLU(), nn.Linear(H, D))

        masks = torch.from_numpy(np.array([[0]*d + [1]*(D-d), [1]*d + [0]*(D-d)] * nchain).astype(np.float32))

        prior = distributions.MultivariateNormal(torch.zeros(D), torch.eye(D))
        self.flow = RealNVP(nets, nett, masks, prior)


        #  Init mixture
        self.pi = np.random.rand(self.n_states, self.n_prob_components)
        normalize(self.pi, axis=1)

        # Init networks
        self.networks = [RealNVP(nets, nett, masks, prior) for _ in range(self.n_prob_components*self.n_states)]

        # Optimizer
        self.optimizer = torch.optim.Adam(sum([[p for p in flow.parameters() if p.requires_grad == True]\
                                            for flow in self.networks], []), lr=1e-4)

        # Reshape in a n_states x n_prob_components array
        self.networks = np.array(self.networks).reshape(self.n_states, self.n_prob_components)
        return self


    def llh(self, X):
        return self._compute_log_likelihood(X)

    def _compute_log_likelihood(self, X):
        """Computes per-component log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------
        logprob : array, shape (n_samples, n_components)
            Log probability of each sample in ``X`` for each of the
            model states.
        """

        n_samples = X.shape[0]
        llh = np.zeros((n_samples, self.n_states))

        self.loglh_sk = self.var_nograd(np.zeros((self.n_states, self.n_prob_components, n_samples)))
        self.logPIk_s = self.var_nograd(self.pi).log()

        X_ = self.dtype(X)

        # One likelihood function per state
        f_s = [partial(self._compute_log_likelihood_per_state, x=X_, s=s)
               for s in range(self.n_states)]

        # For each state
        for i, llh_fun in enumerate(f_s):
            llh[:, i] = llh_fun()
        return llh


    def _generate_sample_from_state(self, state, random_state=None):
        """Generates a random sample from a given component.

        Parameters
        ----------
        state : int
            Index of the component to condition on.

        random_state: RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.

        Returns
        -------
        X : array, shape (n_features, )
            A random sample from the emission distribution corresponding
            to a given component.
        """
        return self.to_numpy(self.networks[state].sample(1))


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
                 'start': np.zeros(self.n_states),
                 'trans': np.zeros((self.n_states, self.n_states)),
                 'mixture': np.zeros((self.n_states, self.n_prob_components)),
                 'loss': self.var_nograd(np.array([0]))
                 }

        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        """Updates sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~base._BaseHMM._initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features)
            Sample sequence.

        framelogprob : array, shape (n_samples, n_components)
            Log-probabilities of each sample under each of the model states.

        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.

        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            Log-forward and log-backward probabilities.
        """

        stats['nobs'] += 1
        stats['nframes'] += X.shape[0]

        if self.em_skip_cond():
            saved_params= self.params
            self.params = 'g'

        n_samples, n_components = framelogprob.shape
        n_samples, n_states = framelogprob.shape

        if 's' in self.params:
            stats['start'] += posteriors[0]

        if 't' in self.params:
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            log_xi_sum = np.full((n_components, n_components), -np.inf)
            _hmmc._compute_log_xi_sum(n_samples, n_components, fwdlattice,
                                      log_mask_zero(self.transmat_),
                                      bwdlattice, framelogprob,
                                      log_xi_sum)
            with np.errstate(under="ignore"):
                stats['trans'] += np.exp(log_xi_sum)

        if 'm' in self.params:

            max_loglh = torch.max(torch.max(self.var_nograd(self.loglh_sk), dim=2)[0], dim=1)[0]
            #max_loglh = 0
            gamma_ = np.zeros((self.n_states, self.n_prob_components, n_samples))

            for i in range(self.n_states):
                    for t in range(n_samples):
                        for m in range(self.n_prob_components):
                            gamma_[i, m, t] = self.pi[i, m] * self.var_nograd(self.loglh_sk - max_loglh[i])[i, m, t].exp()

                        gamma_[i, :, t] /= (gamma_[i,:,t].sum() + 1e-6)   #.reshape(self.n_states,1,n_samples)
                        gamma_[i, :, t] *= posteriors[t, i]

            stats["mixture"] += gamma_.sum(2)

        if 'g' in self.params:
            #  TODO: make sure the graph does not grow during forward and backward.
            # stats["loss"] += self.loss

            # Resize P(chi | S) for broadcasting
            logPIk_s_ext = self.var_nograd(self.logPIk_s.reshape(self.n_states, self.n_prob_components, 1))

            # Brackets = log-P(X | chi, S) + log-P(chi | s)
            brackets = self.loglh_sk + logPIk_s_ext

            # Compute log-p(chi | s, X) = log-P(X|s,chi) + log-P(chi|s) - log\sum_{chi} exp ( log-P(X|s,chi) + log-P(chi|s) )
            log_num = self.loglh_sk.detach() + logPIk_s_ext
            log_denom = self.var_nograd(lsexp(self.loglh_sk.detach() + logPIk_s_ext, axis=1))

            logpk_sX = log_num - log_denom.reshape(self.n_states, 1, n_samples)

            post = self.var_nograd(posteriors.swapaxes(0, 1))  # Transform into n_states x n_samples

            #  The .sum(1) call sums on the components and .sum() sums on all states and samples
            loss = -(post * (torch.exp(logpk_sX) * brackets).sum(1)).sum()/stats['nobs']
            loss.backward()
            stats['loss'] += float(loss.detach().numpy())

        if self.em_skip_cond():
            self.params = saved_params



    def _do_mstep(self, stats):
        """Performs the M-step of EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
        """

        if self.em_skip_cond():
            saved_params= self.params
            self.params = 'g'


        # The ``np.where`` calls guard against updating forbidden states
        # or transitions in e.g. a left-right HMM.
        if 's' in self.params:
            startprob_ = self.startprob_prior - 1.0 + stats['start']
            self.startprob_ = np.where(self.startprob_ == 0.0,
                                       self.startprob_, startprob_)
            normalize(self.startprob_)

        if 't' in self.params:
            transmat_ = self.transmat_prior - 1.0 + stats['trans']
            self.transmat_ = np.where(self.transmat_ == 0.0,
                                      self.transmat_, transmat_)
            normalize(self.transmat_, axis=1)

        if 'm' in self.params:
            self.pi = stats["mixture"]

            # In case we get a line of zeros in the stats
            #self.pi[self.pi.sum(1) == 0, :] = np.ones(self.n_prob_components) / self.n_prob_components
            normalize(self.pi, axis=1)

        if 'g' in self.params:
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(self.monitor_.iter, stats['loss'])
            pass

        if self.em_skip_cond():
            self.params = saved_params


    def init_future(self):
        """
        Initialize dictionary which will store the updates during training.
        """
        self.future = dict([(k, self.__getattribute__(k)) for k in self.model_params])
        return self

    def apply_future(self):
        """
        Apply parameters calculated during a training iteration to the HMM object.
        """
        for attr, value in self.future.items():
            self.__setattr__(attr, value)
        return self

    def _draw_component(self, s):
        """
        Draw component for a given state, based on probability pi[s]

        Parameters
        ----------
        s: int, index of HMM state

        Returns
        -------
        k: int, index of component in state s
        """
        return np.random.choice(np.arange(self.n_states), 1, p=self.pi[s])[0]

    @staticmethod
    def to_numpy(x):
        return x.detach().numpy()

    def var_nograd(self, x):
        return Variable(self.dtype(x), requires_grad=False)

    def var_grad(self, x):
        return Variable(self.dtype(x), requires_grad=True)

    def _compute_log_likelihood_per_state(self, x=None, s=None):
        """Input types must be torch.tensor."""
        # Compute llh per prob model component
        loglh_sk = [self.networks[s, k].log_prob(x).reshape(1, -1)/x.numel() for k in range(self.pi[s].shape[0])]
        #[self.networks[s, k].log_prob(x).reshape(1, -1) for k in range(self.pi[s].shape[0])]
        #[self.networks[s, k].log_prob(x).reshape(1, -1) for k in range(self.pi[s].shape[0])]

        self.loss = torch.cat(loglh_sk, dim=0)

        self.loglh_sk[s] = self.loss

        return self.var_nograd(self.logPIk_s[s].reshape(self.n_prob_components,1) + self.loss).detach().numpy().sum(0)