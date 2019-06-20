import torch


def _logaddexp(a, b):
    output = torch.zeros_like(a)

    out_put_b_mask = torch.isinf(a) & (a < 0)
    out_put_a_mask = torch.isinf(b) & (b < 0)

    output[out_put_b_mask] = b[out_put_b_mask]
    output[out_put_a_mask] = a[out_put_a_mask]
    
    rest_mask = ~(out_put_a_mask | out_put_b_mask)
    
    c = torch.cat((a[None,:], b[None,:]), dim=0)

    output[rest_mask] = torch.logsumexp(c, dim=0)[rest_mask]
    
    return output
    

def _compute_log_xi_sum(n_samples, n_components,
                        fwdlattice,
                        log_transmat,
                        bwdlattice,
                        framelogprob,
                        log_xi_sum):
    """compute the gamma, in order to update transition matrix of hmm"""
    work_buffer = torch.zeros_like(log_transmat)
    logprob = torch.logsumexp(fwdlattice[n_samples - 1], dim=-1)
    
    for t in range(n_samples - 1):
        for i in range(n_components):
            work_buffer[i,:] = fwdlattice[t, i] + \
                               log_transmat[i, :] + \
                               framelogprob[t+1, :] + \
                               bwdlattice[t+1, :] \
                               - logprob

        log_xi_sum = _logaddexp(log_xi_sum, work_buffer)

    return log_xi_sum


def _forward(n_samples, n_components, log_startprob,
             log_transmat, framelogprob):
    """Backward method"""
    fwdlattice = torch.zeros_like(framelogprob)
        
    fwdlattice[0, :] = log_startprob + framelogprob[0, :]
    for t in range(1, n_samples):
        for j in range(n_components):
            work_buffer = fwdlattice[t-1, :] + log_transmat[:, j]

            fwdlattice[t, j] = torch.logsumexp(work_buffer, dim=-1) + framelogprob[t, j]

    #with np.errstate(under="ignore"):
    return torch.logsumexp(fwdlattice[-1], dim=-1), fwdlattice

def _backward(n_samples, n_components, log_startprob,
              log_transmat, framelogprob):
    """Forward method"""
    
    bwdlattice = torch.zeros_like(framelogprob)
    # last row is already zeros, so omit the zero setting step
    for t in range(n_samples - 2, -1, -1):
        for i in range(n_components):
            work_buffer = log_transmat[i,:] + framelogprob[t + 1, :] + bwdlattice[t+1, :]
            bwdlattice[t, i] = torch.logsumexp(work_buffer, dim=-1)
    return bwdlattice

