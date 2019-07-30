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
                 'start': np.zeros(self.n_components),
                 'trans': np.zeros((self.n_components, self.n_components)),
                 'post_comp_mix': 0,
                 'weighted_mean_nomer':0,
                 'cov_numer': 0}
        return stats
    
    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          post_comp, fwdlattice, bwdlattice):

        super(hmm.GMMHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, post_comp, fwdlattice, bwdlattice
        )

        n_samples, _ = X.shape

        stats['n_samples'] = n_samples
        

        prob_mix = np.zeros((n_samples, self.n_components, self.n_mix))
        for p in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, p)
            with np.errstate(under="ignore"):
                prob_mix[:, p, :] = np.exp(log_denses) + np.finfo(np.float).eps

        prob_mix_sum = np.sum(prob_mix, axis=2)
        post_mix = prob_mix / prob_mix_sum[:, :, np.newaxis]
        post_comp_mix = post_comp[:, :, np.newaxis] * post_mix
        stats['post_comp_mix'] += post_comp_mix.sum(axis=0)
        
        stats['weighted_mean_nomer'] += np.matmul(post_comp_mix.transpose(1, 2, 0), X)
        
        # stats['post_mix_sum'] = np.sum(post_comp_mix, axis=0)
        # stats['post_sum'] = np.sum(post_comp, axis=0)

        centered = X[:, np.newaxis, np.newaxis, :] - self.means_
        centered2 = centered ** 2
        stats["cov_numer"] = np.einsum('ijk,ijkl->jkl',\
                                       post_comp_mix, centered2)
        
        
    def _do_mstep(self, stats):
        super(hmm.GMMHMM, self)._do_mstep(stats)

        n_samples = stats['n_samples']
        n_features = self.n_features

        # Maximizing weights
        
        new_weights = stats['post_comp_mix']
        normalize(new_weights, axis=1)
        
        # Maximizing means

        new_means_numer = stats['weighted_mean_nomer']
        new_means_denom = stats['post_comp_mix']
        try:
            
            new_means = new_means_numer / (new_means_denom[:,:,None] + 1e-6)
        except:
            print('error')

        # ONLY diag case of co-variance is implemented now. Maximizing covariance
        

        if self.covariance_type == 'full':
            # copy codo, not adapted yet
            centered = stats['centered'].reshape((
                n_samples, self.n_components, self.n_mix, self.n_features, 1
            ))
            centered_t = stats['centered'].reshape((
                n_samples, self.n_components, self.n_mix, 1, self.n_features
            ))
            centered_dots = centered * centered_t

            psis_t = np.transpose(self.covars_prior, axes=(0, 1, 3, 2))
            nus = self.covars_weight

            centr_means_resh = centered_means.reshape((
                self.n_components, self.n_mix, self.n_features, 1
            ))
            centr_means_resh_t = centered_means.reshape((
                self.n_components, self.n_mix, 1, self.n_features
            ))
            centered_means_dots = centr_means_resh * centr_means_resh_t

            new_cov_numer = np.einsum(
                'ijk,ijklm->jklm',
                stats['post_comp_mix'], centered_dots
            ) + psis_t + (lambdas[:, :, np.newaxis, np.newaxis] *
                          centered_means_dots)
            new_cov_denom = (
                stats['post_mix_sum'] + 1 + nus + self.n_features + 1
            )[:, :, np.newaxis, np.newaxis]

            new_cov = new_cov_numer / new_cov_denom
        elif self.covariance_type == 'diag':
            # current working case
            new_cov_numer = stats["cov_numer"]
            new_cov_denom = stats["post_comp_mix"]

            new_cov = new_cov_numer / (new_cov_denom[:,:,None] + 1e-6)
        elif self.covariance_type == 'spherical':
            # copy code, not imported yet
            centered_norm2 = np.sum(stats['centered'] ** 2, axis=-1)

            alphas = self.covars_prior
            betas = self.covars_weight

            centered_means_norm2 = np.sum(centered_means ** 2, axis=-1)

            new_cov_numer = np.einsum(
                'ijk,ijk->jk',
                stats['post_comp_mix'], centered_norm2
            ) + lambdas * centered_means_norm2 + 2 * betas
            new_cov_denom = (
                n_features * stats['post_mix_sum'] + n_features +
                2 * (alphas + 1)
            )

            new_cov = new_cov_numer / new_cov_denom
        elif self.covariance_type == 'tied':
            centered = stats['centered'].reshape((
                n_samples, self.n_components, self.n_mix, self.n_features, 1
            ))
            centered_t = stats['centered'].reshape((
                n_samples, self.n_components, self.n_mix, 1, self.n_features
            ))
            centered_dots = centered * centered_t

            psis_t = np.transpose(self.covars_prior, axes=(0, 2, 1))
            nus = self.covars_weight

            centr_means_resh = centered_means.reshape((
                self.n_components, self.n_mix, self.n_features, 1
            ))
            centr_means_resh_t = centered_means.reshape((
                self.n_components, self.n_mix, 1, self.n_features
            ))
            centered_means_dots = centr_means_resh * centr_means_resh_t

            lambdas_cmdots_prod_sum = np.einsum(
                'ij,ijkl->ikl',
                lambdas, centered_means_dots
            )

            new_cov_numer = np.einsum(
                'ijk,ijklm->jlm',
                stats['post_comp_mix'], centered_dots
            ) + lambdas_cmdots_prod_sum + psis_t
            new_cov_denom = (
                stats['post_sum'] + self.n_mix + nus + self.n_features + 1
            )[:, np.newaxis, np.newaxis]

            new_cov = new_cov_numer / new_cov_denom

        # Assigning new values to class members
        self.weights_ = new_weights
        self.means_ = new_means
        self.covars_ = new_cov
