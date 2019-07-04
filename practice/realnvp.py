import numpy as np
import matplotlib.pyplot as plt
#% matplotlib inline

from pylab import rcParams

rcParams['figure.figsize'] = 10, 8
rcParams['figure.dpi'] = 300

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from src.realnvp import RealNVP

if __name__ == "__main__":
    H = 28
    d = 2
    nchain = 3
    niter = 5000
    n_samples_train = 100
    n_samples_test = 1000

    nets = lambda: nn.Sequential(nn.Linear(d, H), nn.LeakyReLU(), nn.Linear(H, H), nn.LeakyReLU(), nn.Linear(H, d),
                                 nn.Tanh())
    nett = lambda: nn.Sequential(nn.Linear(d, H), nn.LeakyReLU(), nn.Linear(H, H), nn.LeakyReLU(), nn.Linear(H, d))

    masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * nchain).astype(np.float32))
    prior = distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
    flow = RealNVP(nets, nett, masks, prior)

    optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=1e-4)

    for t in range(niter):
        noisy_moons = datasets.make_moons(n_samples=n_samples_train, noise=.05)[0].astype(np.float32)
        x = torch.from_numpy(noisy_moons).reshape(n_samples_train, 1, d)
        mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.uint8)
        loss = -flow.log_prob(x, mask).mean()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if t % 500 == 0:
            print('iter %s:' % t, 'loss = %.3f' % loss)


    noisy_moons = datasets.make_moons(n_samples=n_samples_test, noise=.05)[0].astype(np.float32)
    xtest = torch.from_numpy(noisy_moons).reshape(n_samples_test, 1, d)
    mask = torch.ones((xtest.shape[0], xtest.shape[1]), dtype=torch.uint8)

    z = flow.f(xtest)[0].detach().numpy().squeeze()
    plt.subplot(221)
    plt.scatter(z[:, 0], z[:, 1])
    plt.title(r'$z = f(X)$')

    z = np.random.multivariate_normal(np.zeros(2), np.eye(2), n_samples_test)
    plt.subplot(222)
    plt.scatter(z[:, 0], z[:, 1])
    plt.title(r'$z \sim p(z)$')

    plt.subplot(223)
    x = datasets.make_moons(n_samples=n_samples_test, noise=.05)[0].astype(np.float32)
    plt.scatter(x[:, 0], x[:, 1], c='r')
    plt.title(r'$X \sim p(X)$')

    plt.subplot(224)
    x = flow.sample(n_samples_test).detach().numpy()
    plt.scatter(x[:, 0, 0], x[:, 0, 1], c='r')
    plt.title(r'$X = g(z)$')

    print("")