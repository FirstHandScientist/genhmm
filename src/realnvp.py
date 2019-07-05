import torch


class RealNVP(torch.nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = torch.nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def g(self, z):
        # not sure about this
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros((x.shape[0], x.shape[1])), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z + t) * torch.exp(s) + z_
            log_det_J += s.sum(dim=2)
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
