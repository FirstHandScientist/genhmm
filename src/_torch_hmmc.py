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
