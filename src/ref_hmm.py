import sys
import numpy as np
import pickle as pkl
from hmmlearn.base import ConvergenceMonitor
from hmmlearn.utils import iter_from_X_lengths, normalize
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
        if int(self.iepoch) == 1:
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
    def __init__(self, *args, **kwargs):
        super(GMM_HMM, self).__init__(*args, **kwargs)
        return

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
        if int(self.iepoch) == 1:
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
                # if curr_logprob < 0:
                #     print("negative log likelihood")

                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                stats = self._accumulate_sufficient_statistics(
                    stats, X[i:j], framelogprob, posteriors, fwdlattice,
                    bwdlattice)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats,X)
            
            self.monitor_.report(curr_logprob/stats["nobs"])
            # if self.monitor_.converged:
            #     break

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
                 'n_samples': 0,
                 'start': np.zeros(self.n_components),
                 'trans': np.zeros((self.n_components, self.n_components)),
                 'samples':[],
                 'post_comp_mix': [],
                 'post_mix_sum': [],
                 'post_sum': []}
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          post_comp, fwdlattice, bwdlattice):

        super(hmm.GMMHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, post_comp, fwdlattice, bwdlattice
        )

        n_samples, _ = X.shape
        stats['n_samples'] += n_samples
        stats['samples'].append(X)

        prob_mix = np.zeros((n_samples, self.n_components, self.n_mix))
        for p in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, p)
            with np.errstate(under="ignore"):
                prob_mix[:, p, :] = np.exp(log_denses) + np.finfo(np.float).eps

        prob_mix_sum = np.sum(prob_mix, axis=2)
        post_mix = prob_mix / prob_mix_sum[:, :, np.newaxis]
        post_comp_mix = post_comp[:, :, np.newaxis] * post_mix

        stats['post_comp_mix'].append(post_comp_mix)
        stats['post_mix_sum'].append(np.sum(post_comp_mix, axis=0))
        stats['post_sum'].append(np.sum(post_comp, axis=0))
        return stats

    def _do_mstep(self, stats, X):
        super(hmm.GMMHMM, self)._do_mstep(stats)
        post_mix_sum = np.concatenate([x[..., np.newaxis] for x in stats["post_mix_sum"]], axis=-1)
        post_sum = np.concatenate([x[..., np.newaxis] for x in stats["post_sum"]], axis=-1)
        post_comp_mix = np.concatenate(stats["post_comp_mix"], axis=0)


        # Maximizing weights
        alphas_minus_one = self.weights_prior - 1
        new_weights_numer = post_mix_sum + alphas_minus_one[...,np.newaxis]
        new_weights_denom = (
            post_sum + np.sum(alphas_minus_one, axis=1)[..., np.newaxis]
        )[:, np.newaxis, :]
        new_weights = (new_weights_numer / new_weights_denom).sum(-1) / stats["nobs"]


        # Maximizing means
        lambdas, mus = self.means_weight, self.means_prior
        new_means_numer = np.einsum(
            'ijk,il->jkl',
            post_comp_mix, X
        ) + lambdas[:, :, np.newaxis] * mus
        new_means_denom = (post_mix_sum.sum(-1) + lambdas)[:, :, np.newaxis]
        new_means = new_means_numer / new_means_denom


        # Maximizing cov
        centered_means = self.means_ - mus

        centered2 = (X[:, np.newaxis, np.newaxis, :] - self.means_) ** 2
        centered_means2 = centered_means ** 2

        alphas = self.covars_prior
        betas = self.covars_weight

        new_cov_numer = np.einsum(
            'ijk,ijkl->jkl',
            post_comp_mix, centered2
        ) + lambdas[:, :, np.newaxis] * centered_means2 + 2 * betas
        new_cov_denom = (
                post_mix_sum.sum(-1)[:, :, np.newaxis] + 1 + 2 * (alphas + 1)
        )

        new_cov = new_cov_numer / new_cov_denom

        # Assigning new values to class members
        self.weights_ = new_weights
        self.means_ = new_means
        self.covars_ = new_cov
