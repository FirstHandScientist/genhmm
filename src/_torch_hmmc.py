import torch
import sys


def _logaddexp(a, b, mask):
    """compute the log sum exp, to avoid the situation of add to inf values"""
    output = torch.zeros_like(a)
    # find the mask to output b when a contain -inf values
    out_put_b_mask = torch.isinf(a) & (a < 0)

    # find the mask to output a when b contain -inf values
    out_put_a_mask = torch.isinf(b) & (b < 0)
    # in order not to take the padded number into account
    # stop do accumulating when iteration gets in padded data
    out_put_a_mask = out_put_a_mask | ~ mask[:, None, None]

    # if no singularity cases happen, set the masks for logsumexp computations
    rest_mask = ~(out_put_a_mask | out_put_b_mask)

    # set value for found masks
    output[out_put_b_mask] = b[out_put_b_mask]
    output[out_put_a_mask] = a[out_put_a_mask]
    c = torch.cat((a[None,:], b[None,:]), dim=0)
    output[rest_mask] = torch.logsumexp(c, dim=0)[rest_mask]
    
    return output
    

def _compute_log_xi_sum(n_samples, n_components, fwdlattice, \
                        log_transmat, bwdlattice, batch_framelogprob, \
                        log_xi_sum, logprob, mask):
    """Compute the gamma, in order to update transition matrix of hmm
    INPUT: 
    n_samples: time step length of padded sequence, including padded zeros, i.e. max sequence length
    n_components: number of hmm states

    fwdlattice: batch fwdlattice. shape: batch_size * max_sequence_length. See output in method of _forward
    
    log_transmat: the log transition probability matrix of hmm, 
                  shape: number of hmm states X number of hmm states

    bwdlattice: the batch bwdlattice, Shape: batch_size * max_sequence_length * number of hmm states
                See output explanation in output format of _backward
        

    batch_framelogprob: batch loglikelihood, the padded positions should be set as zeros
                  shape: batch_size * max sequence length * number of hmm states
                  See example of _forward method input explanation 
    
    log_xi_sum: this input should be initialized as :
                log_xi_sum = torch.ones(batch_size, n_components, n_components)*float('-inf')
    
    logprob: the loglikelihood of sequences in batch data. Shape: 1 * batch_size. The first output of _forward method.

    mask: the mask of valid data in batch_framelogprob
          shape: batch_size * max sequence length
          example of mask for the above batch_framelogprob:
            [[1 1 1 0 0 0 0 0 0],
             [1 1 1 1 1 0 0 0 0]]
    
    OUTPUT:
    log_xi_sum: the update log_xi_sum
    """

    batch_size=batch_framelogprob.shape[0]
    work_buffer = torch.zeros((batch_size, \
                               log_transmat.shape[0], \
                               log_transmat.shape[1]), \
                              device=mask.device)
    log_transmat = log_transmat.reshape(1,n_components,n_components).repeat(batch_size,1,1)
    
    
    for t in range(n_samples - 1):
        for i in range(n_components):
            work_buffer[:, i,:] = fwdlattice[:, t, i].reshape(-1, 1) + \
                               log_transmat[:, i, :] + \
                               batch_framelogprob[:, t+1, :] + \
                               bwdlattice[:, t+1, :] \
                               - logprob.reshape(-1,1)

        log_xi_sum = _logaddexp(log_xi_sum, work_buffer, mask[:,t+1])  

    return log_xi_sum


def _forward(n_samples, n_components, log_startprob,
             log_transmat, batch_framelogprob, mask):
    """Forward method implementation for batch 

    INPUT:
    n_samples: time step length of padded sequence, including padded zeros, i.e. max sequence length
    n_components: number of hmm states
    log_startprob: the startprob vector of hmm model, 
                  shape: 1 X number of hmm states
    
    log_transmat: the log transition probability matrix of hmm, 
                  shape: number of hmm states X number of hmm states

    batch_framelogprob: batch loglikelihood, the padded positions should be set as zeros
                  shape: batch_size * max sequence length * number of hmm states
                example of a batch_framelogprob with batch_size 2, max_sequence_length 9, number of hmm states 4
                [[[-1.82216701 -0.02628716 -2.52146999 -0.20764606]
                  [-0.7557014  -0.37549729 -1.70601169 -1.40064122]
                  [-1.16290289 -0.36987039 -0.39243632 -0.76830808]
                  [ 0.          0.          0.          0.        ]
                  [ 0.          0.          0.          0.        ]
                  [ 0.          0.          0.          0.        ]
                  [ 0.          0.          0.          0.        ]
                  [ 0.          0.          0.          0.        ]
                  [ 0.          0.          0.          0.        ]]

                 [[-1.38754713 -0.27324662 -1.30298724 -0.84129666]
                  [-2.44544926 -0.24396847 -0.29964475 -0.37567767]
                  [-1.01470265 -0.53463464 -0.68245776 -0.47579989]
                  [-1.01754508 -0.55297899 -1.32827442 -1.31774117]
                  [-0.36268008 -0.52209605 -0.83685421 -0.2940473 ]
                  [ 0.          0.          0.          0.        ]
                  [ 0.          0.          0.          0.        ]
                  [ 0.          0.          0.          0.        ]
                  [ 0.          0.          0.          0.        ]]]

    mask: the mask of valid data in batch_framelogprob
          shape: batch_size * max sequence length
          example of mask for the above batch_framelogprob:
            [[1 1 1 0 0 0 0 0 0],
             [1 1 1 1 1 0 0 0 0]]

    OUTPUT:
    batch_logprob: batch loglikelihood of sequence given hmm (only valid data are considered, excluding padding zeros)
                   shape: 1 * batch_size  
                   For instance: [-2.9063, -4.1796]
    fwdlattice: batch fwdlattice. shape: batch_size * max_sequence_length
        example of batch fwdlattice:
        tensor([[[-2.5756, -1.1117, -5.2945, -2.2557],
                 [-3.6001, -2.7751, -4.2645, -3.7963],
                 [-4.9337, -3.9152, -4.1365, -4.4599],
                 [ 0.0000,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000]],

                [[-2.1410, -1.3586, -4.0760, -2.8893],
                 [-5.0254, -2.3800, -2.8705, -3.0144],
                 [-4.2677, -4.4636, -3.9802, -3.3246],
                 [-5.1304, -4.7781, -5.1226, -5.0957],
                 [-5.4590, -5.6160, -5.9375, -5.3456],
                 [ 0.0000,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000]]], dtype=torch.float64)
    """

    batch_size = batch_framelogprob.shape[0]
    fwdlattice = torch.zeros_like(batch_framelogprob)
            
    fwdlattice[:, 0, :] = log_startprob + batch_framelogprob[:,0, :]
    for t in range(1, n_samples):
        for j in range(n_components):
            work_buffer = fwdlattice[:, t-1, :] + log_transmat[:, j]
            # fwdlattice[:,t, j] = torch.logsumexp(work_buffer, dim=-1) + \
            #                      framelogprob[:,t, j] 
    
            fwdlattice[:,t, j] = torch.logsumexp(work_buffer, dim=-1) * \
                                 mask[:, t].type(batch_framelogprob.dtype) + batch_framelogprob[:,t, j] 

    # need to find the idx for last sample of each sequence by mask.sum(dim=-1)-1, fwdlattice[:, -1, :] would give the padded zeros
    batch_logprob = torch.logsumexp(fwdlattice[list(range(batch_size)), mask.sum(dim=-1)-1, :], dim=-1)
    return batch_logprob, fwdlattice

def _backward(n_samples, n_components, log_startprob,
              log_transmat, framelogprob, mask):
    """Backward method, implemented in batch
    INPUT: format is same as the input of _forward method

    OUTPUT: bwdlattice. Shape: batch_size * max_sequence_length * number of hmm states
            Example of output batch bwdlattice with batch_size 2, max_sequence_length 9, number of hmm states 4
            and mask
                    [[1 1 1 0 0 0 0 0 0],
                     [1 1 1 1 1 0 0 0 0]]

            tensor([[[-1.1161, -2.7773, -1.1539, -2.6112],
                 [-0.0539, -1.7095, -0.4676, -0.9476],
                 [ 0.0000, -0.0000,  0.0000, -0.0000],
                 [ 0.0000, -0.0000,  0.0000, -0.0000],
                 [ 0.0000, -0.0000,  0.0000, -0.0000],
                 [ 0.0000, -0.0000,  0.0000, -0.0000],
                 [ 0.0000, -0.0000,  0.0000, -0.0000],
                 [ 0.0000, -0.0000,  0.0000, -0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000]],

                [[-2.7500, -4.1936, -2.8118, -2.9566],
                 [-1.9756, -3.5268, -1.9623, -2.5826],
                 [-1.0792, -2.7777, -1.2028, -2.1448],
                 [-0.0527, -1.4457, -0.1206, -0.9245],
                 [ 0.0000, -0.0000,  0.0000, -0.0000],
                 [ 0.0000, -0.0000,  0.0000, -0.0000],
                 [ 0.0000, -0.0000,  0.0000, -0.0000],
                 [ 0.0000, -0.0000,  0.0000, -0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000]]], dtype=torch.float64)
    """
    
    bwdlattice = torch.zeros_like(framelogprob)
    # last row is already zeros, so omit the zero setting step
    for t in range(n_samples - 2, -1, -1):
        for i in range(n_components):
            work_buffer = log_transmat[i,:] + framelogprob[:,t + 1, :] + bwdlattice[:,t+1, :]
            bwdlattice[:, t, i] = torch.logsumexp(work_buffer, dim=-1) * mask[:, t+1].type(framelogprob.dtype)
    return bwdlattice

