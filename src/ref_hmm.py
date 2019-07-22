import sys
import numpy as np
import pickle as pkl
from hmmlearn.base import ConvergenceMonitor

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
