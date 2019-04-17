


import os,sys
import numpy as np

import torch
import matplotlib.pyplot as plt
from torch import nn

from torch.autograd import Variable
from functools import partial



def plotfun(x,y=None, h=None, **kwargs):
    if h is None:
        plt.figure()

    if y is None:
        plt.scatter(x[:, 0], x[:, 1], **kwargs)
    else:
        plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], cmap='coolwarm', **kwargs)



def gendata(n, snr=+np.inf):
    d = 2
    x = np.random.rand(n, d)*2 - 1
    l2 = np.sqrt((x**2).sum(1)).reshape(-1, 1) @ np.ones((1,2))
    x = x / l2
    y = np.concatenate(((x[:, 1] < 0).reshape(-1, 1), ~(x[:, 1] < 0).reshape(-1, 1)), axis=1)

    x[y[:, 0]] += 0.6

    # add noise
    #snr = 15
    sigma_n = x.std() / snr

    x += np.random.randn(x.shape[0],x.shape[1])*sigma_n
    return x, y

if __name__ == "__main__":
    n = 1000
    D = 2
    x, y = gendata(n, snr=15)

    dtype = torch.FloatTensor

    # plotfun(x, y)
    # plt.show()

    # INflate
    #X = torch.FloatTensor(np.concatenate((x, x+3), axis=1))
    #D = 4

    X = dtype(x)

    # split
    d = 1
    H = 10

    w11 = Variable(torch.randn(d, H).type(dtype), requires_grad=True)
    w21 = Variable(torch.randn(H, d).type(dtype), requires_grad=True)

    w12 = Variable(torch.randn(d, H).type(dtype), requires_grad=True)
    w22 = Variable(torch.randn(H, d).type(dtype), requires_grad=True)

    s = Variable(torch.ones(d+d), requires_grad=True)


    x1 = X[:, :d]
    x2 = X[:, D-d:]

    def m(x, w1=None, w2=None):
        return x.mm(w1).clamp(min=0).mm(w2)

    m1 = partial(m, w1=w11, w2=w21)
    m2 = partial(m, w1=w12, w2=w22)
    ########
    # Additive coupling function

    def g(x1, x2, m):
        #logdet = torch.FloatTensor(np.array([1.])).log()
        logdet = m(x1).abs().log().sum()
        return x2 * m(x1), logdet


    def f(x1, x2, s, g, m1,m2):
        y1_1 = x1
        y2_1, logdet1 = g(x1, x2, m1)

        y1_2, logdet2 = g(y2_1, y1_1, m2)
        y2_2 = y2_1

        y1 = y1_2 * s[0]
        y2 = y2_2 * s[1]
        logdet = logdet1 + logdet2 + s.abs().log().sum()
        return y1, y2, logdet
    #########

    # Inverse
    def f_i(y1, y2, s, g_i, m1,m2):
        y1_2 = y1 / s[0]
        y2_2 = y2 / s[1]

        y2_1 = y2_2
        y1_1 = g_i(y2_1, y1_2, m2)

        xr1 = y1_1
        xr2 = g_i(xr1, y2_1, m1)

        return xr1, xr2   #  g_i(xh1, xh2, m, w1, w2)


    def g_i(xh1, xh2, m):
        return xh2 / m(xh1)


    # Logistic Distribution Prior
    def llpH(z):
        return -((1 + z.exp()).log() + (1 + (-z).exp()).log()).sum()

    # Standard gaussian priors
    def llgpH(z):
        log2pi = torch.FloatTensor(np.array([2*np.pi])).log()
        return -1/2*(z.pow(2) + log2pi).sum()

    # Compute llh

    learning_rate = 1e-4
    for t in range(20):
        # h.clf()
        # plotfun(x, y, h=h, alpha=.1)
        # Forward pass
        y1, y2, logdet = f(x1, x2, s, g, m1, m2)

        z = torch.cat((y1, y2), 1)

        llh = llgpH(z) + logdet

        llh.backward()
        llh_val = llh.detach().numpy().tolist()

        if t % 2 == 0:
            print(t, llh_val)

        if np.isnan(llh_val):
            break

        w11.data += learning_rate * w11.grad.data
        w21.data += learning_rate * w21.grad.data
        w12.data += learning_rate * w12.grad.data
        w22.data += learning_rate * w22.grad.data

        s.data += learning_rate * s.grad.data

        w11.grad.data.zero_()
        w21.grad.data.zero_()
        w12.grad.data.zero_()
        w22.grad.data.zero_()

        s.grad.data.zero_()


        plt.pause(1)

    nsample = 1000

    #h = dtype(np.random.randn(nsample, 2))
    h1 = z[:, :d]
    h2 = z[:, D - d:]

    xhat1, xhat2 = f_i(h1, h2, s, g_i, m1, m2)
    f_ix = torch.cat((xhat1, xhat2), 1).detach().numpy()

    fig = plt.figure()
    plotfun(x, y, h=fig, alpha=.1, label="x")
    plotfun(z.detach().numpy(), y, h=fig, label="z ~ p_Z")
    plotfun(f_ix, h=fig, alpha=.5, marker="+", c='black', label="x=f^{-1}(z)")
    fig.legend()
    plt.pause(0.001)

    #ax.lines.pop(1)


    # Sampling


    plotfun(z.detach().numpy(), y)


    print("")




