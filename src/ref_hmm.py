import sys
import numpy as np
import pickle as pkl
from hmmlearn.base import ConvergenceMonitor
from hmmlearn.utils import iter_from_X_lengths
from sklearn.utils import check_array

from hmmlearn import hmm

class ConvgMonitor(ConvergenceMonitor):
    def report(self, logprob):
        """Reports convergence to :data:`sys.stderr`.

        The output consists of three columns: iteration number, log
        probability of the data at the current iteration and convergence
        rate.  At the first iteration convergence rate is unknown and
        is thus denoted by NaN.

        Parameters
        ----------
        logprob : float
            The log probability of the data as computed by EM algorithm
            in the current iteration.
        """
        if self.verbose:
            delta = logprob - self.history[-1] if self.history else np.nan
            message = self._template.format(
                iter=self.iter + 1, logprob=logprob, delta=delta)
            print(message, file=sys.stdout)

        self.history.append(logprob)
        self.iter += 1

class GaussianHMMclassifier():
    def __init__(self, mdlc_files=None, **options):
        """Initialize a model on CPU. Make sure to push to GPU at runtime."""
        super(GaussianHMMclassifier, self).__init__()

        if mdlc_files == None:
            pass
            # self.nclasses = options["nclasses"]
            # self.hmms = [GenHMM(**options) for _ in range(self.nclasses)]
            

        else:
            self.hmms = [pkl.load(open(fname, 'rb')) for fname in mdlc_files]
        

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
            llh = [classHMM.pred_score(x) / classHMM.latestNLL for classHMM in self.hmms]
        else:
            llh = [classHMM.score(x) for classHMM in self.hmms]

        return llh
class Gaussian_HMM(hmm.GaussianHMM):
    def fit(self, X, lengths=None):
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        if self.iepoch == '1':
            self._init(X, lengths=lengths)
            self.monitor_._reset()
        self._check()
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for i, j in iter_from_X_lengths(X, lengths):
                framelogprob = self._compute_log_likelihood(X[i:j])
                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_sufficient_statistics(
                    stats, X[i:j], framelogprob, posteriors, fwdlattice,
                    bwdlattice)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)
            
            self.monitor_.report(curr_logprob)
            # if self.monitor_.converged:
            #     break

        return self
    
class GMM_HMM(hmm.GMMHMM):
    def fit(self, X, lengths=None):
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        if self.iepoch == '1':
            self._init(X, lengths=lengths)
            self.monitor_._reset()
        self._check()
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for i, j in iter_from_X_lengths(X, lengths):
                framelogprob = self._compute_log_likelihood(X[i:j])
                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_sufficient_statistics(
                    stats, X[i:j], framelogprob, posteriors, fwdlattice,
                    bwdlattice)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)
            
            self.monitor_.report(curr_logprob)
            # if self.monitor_.converged:
            #     break

        return self
    
