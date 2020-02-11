import torch

class Rescale(torch.nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, 1, num_channels))

    def forward(self, x):
        x = self.weight * x
        return x

class RealNVP(torch.nn.Module):
    """
    RealNVP module.
    Adapted from https://github.com/senya-ashukha/real-nvp-pytorch
    """
    def __init__(self, nets, mask, prior):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        self.mask = torch.nn.Parameter(mask, requires_grad=False)
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])
        self.rescale = torch.nn.utils.weight_norm(Rescale(int(self.mask.size(1)/2)))
    
    def _chunk(self, x, mask):
        """chunk the input x into two chunks along dimension 2
        INPUT: tensor to be chunked, shape: batch_size * n_samples * n_features
        OUTPUT: tow chunks of tensors with equal size
        """
        idx_id = torch.nonzero(mask).reshape(-1)
        idx_scale = torch.nonzero(~mask).reshape(-1)
        chunk_id = torch.index_select(x, dim=2,
                                      index=idx_id)
        chunk_scale = torch.index_select(x, dim=2,
                                         index=idx_scale)
        return (chunk_id, chunk_scale)
        
    def g(self, z):
        # not sure about this
        x = z
        # for i in range(len(self.t)):
        #     x_ = x * self.mask[i]
        #     s = self.s[i](x_) * (1 - self.mask[i])
        #     t = self.t[i](x_) * (1 - self.mask[i])
        #     x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        # return x
        pass

    def f(self, x):
        log_det_J, z = x.new_zeros((x.shape[0], x.shape[1])), x
        for i in reversed(range(len(self.s))):
            z_id, z_s = self._chunk(z, self.mask[i])
            
            st = self.s[i](z_id)
            s, t = st.chunk(2,dim=2)
            s = self.rescale(torch.tanh(s))
            
            exp_s = s.exp()
            z_s = (z_s + t) * exp_s
            z =  torch.cat((z_id, z_s), dim=2)
            
            log_det_J += torch.sum(s, dim=2)
        return z, log_det_J

    def log_prob(self, x, mask):
        """The prior log_prob may need be implemented such it adapts cuda computation."""
        z, logp = self.f(x)

        px = self.prior.log_prob(z) + logp
        # set the padded positions as zeros
        px[~mask] = 0
        # px[~mask].zero_()
        #if (px > 0).any():
          #  print("here")
        return px

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x
