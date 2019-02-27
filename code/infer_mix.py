"""Train script.
Usage:
    infer.py <hparams> <dataset_root> <mode>
"""
import os
import cv2
import random
import torch
import torchvision.datasets as dset
import torch
import numpy as np
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.config import JsonConfig
from glow import modules
import torchvision.utils as vutils
from scipy import stats


def prior_sampler(pk, size=64):
    """ Define the sample generator for mixture distribution
    xk = np.arrange(number of components)
    pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2), tuple
    R = custm.rvs(size=100)
    """
    xk = np.arange(len(pk))
    custm = stats.rv_discrete(name='custm', values=(xk, pk))
    prior_samples  = custm.rvs(size=size)
    return prior_samples

def GenMMmap_encoding(graph, x):
    z, nll = graph(x=x, y_onehot=None, reverse=False)
       
    nlog_joint_prob =nll - torch.log(pk.unsqueeze(1).expand_as(nll)+1e-6).cuda()
    tmp_sum = torch.log( torch.sum( torch.exp(-nlog_joint_prob), dim=[0]) ).cuda
    nlog_gamma = nlog_joint_prob + tmp_sum().expand_as(nlog_joint_prob)
    _, imin = nlog_gamma.min(dim=0)
    r_c_index = torch.stack([imin.data.cpu().type(torch.int64), torch.arange(0,64).type(torch.int64)])
    
    return z[r_c_index[0], r_c_index[1]], r_c_index

def LatMMmap_encoding(graph, x):
    graph = graph.get_component()
    z, _, _ = graph(x=x, y_onehot=None, reverse=False)
       
    return z

def interpolation(start, end, points=9):
    weight = np.linspace(0.1, 0.9, points)
    inter = []
    for i in weight:
        inter.append(torch.lerp(start, end, i))
    return inter

if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    mode = args["<mode>"]
    assert mode in ["Generating", "Interpolation"]

    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    
    hparams = JsonConfig(hparams)
    batch_size = hparams.Train.batch_size
    builded = build(hparams, False)
    graph = builded["graph"]
    # obtain current prior of each component in the mixture distribution
    #################  1. do the generating work #####################
    pk = builded["graph_prior"]
    IMG_NAME = "GenMM_K{}".format(hparams.Mixture.num_component) if hparams.Mixture.naive else "LatMM_K{}".format(hparams.Mixture.num_component)

    if mode == "Generating":
        # for the_graph in graph:
        #     pk.append(the_graph['prior'])
        pknp = pk.numpy()
        pk = tuple(pk.numpy())
        print("The current model prior is: {}".format(pk))
        prior_samples = prior_sampler(pk, size=batch_size)
        print("[Generator ID: {}]".format(prior_samples) )
        images = []
        images = graph(z=None, eps_std=1, reverse=True)
        images=images.mul(0.5).add(0.5)
        vutils.save_image(images.data.cpu(), "pictures/gen_{}_std1.png".format(IMG_NAME))
        images = graph(z=None, eps_std=1, reverse=True)
        images=images.mul(0.5).add(0.5)
        images = graph(z=None, eps_std=0.8, reverse=True)
        images=images.mul(0.5).add(0.5)
        vutils.save_image(images.data.cpu(), "pictures/gen_{}_std0_8.png".format(IMG_NAME))
    ################  ###############
    if mode == "Interpolation":
        transform = transforms.Compose([
            transforms.CenterCrop(hparams.Data.center_crop),
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = dset.MNIST(root=dataset_root,
                             download=True,
                             transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size*2,
                                             shuffle=True, num_workers=int(2))

        #dataset = dataset(dataset_root, transform=transform)
        batch = next(iter(dataloader))
    ################ 2. do the interpolation for GenMM ###############
    if mode == "Interpolation" and hparams.Mixture.naive:
        x = batch[0]
        y = batch[1]
        digits = list(range(10))
        factor = 2
        x_ordered1 = []
        x_ordered2 = []
        for num in digits:
            assert x[y==num].size(0)>1
            x_ordered1.append(x[y==num][0])
            x_ordered2.append(x[y==num][-1])
        x_ordered = x_ordered1 + x_ordered2
        newbatch = next(iter(dataloader))
        # file x to make its batch size as 64
        for i in range(factor*len(digits), batch_size):
            x_ordered.append(newbatch[0][i])
        x_ordered = torch.stack(x_ordered)
        vutils.save_image(x_ordered, "pictures/sample.png", nrow=10)
        sel_z, rc_index = GenMMmap_encoding(graph=graph, x=x_ordered.cuda())
        #assert False
    
        x_start = x_ordered[:10]
        z_start = sel_z[:10]
        id_start = rc_index[0, :10]
        z_end = sel_z[10:20]
        x_end = x_ordered[10:20]
        id_end = rc_index[0, 10:20]
        points = 8
        weight = np.linspace(0.1, 0.9, points)
        z_iep = []
        gid = []
        for i in weight:
            for j,_ in enumerate(z_start):
                z_iep.append(torch.lerp(z_start[j], z_end[j], i))
                if i <0.5:
                    gid.append(id_start[j])
                else:
                    gid.append(id_end[j])
        # use selected generators to generate sample 
        imgs = []
        # put the starting samples into the list
        for i in range(10):
            imgs.append(x_start[i].data)
        ###### set to false for random generator selection
        if True:
            #### do the generation with chosen generator id by MAP
            for i, idg in enumerate(gid):
                tmp_img = graph.get_component(idg)(z=z_iep[i].expand(batch_size, z_iep[0].size(0), z_iep[0].size(1), z_iep[0].size(2)), reverse= True)
                imgs.append(tmp_img[0].data.cpu())

            # put the ending samples into list
            for i in range(10):
                imgs.append(x_end[i].data)
            # saving
            vutils.save_image(imgs, "pictures/interpolation_{}_map.png".format(IMG_NAME), nrow=10)
        else:
            #### do the generating with random generator id sampled by prior of generator
            pk = tuple(pk.numpy())
            print("The current model prior is: {}".format(pk))
            prior_samples = prior_sampler(pk, size=len(gid))
            print("[Generator ID: {}]".format(prior_samples) )
            for i, idg in enumerate(prior_samples):
                tmp_img = graph.get_component(idg)(z=z_iep[i].expand(batch_size, z_iep[0].size(0), z_iep[0].size(1), z_iep[0].size(2)), reverse= True)
                imgs.append(tmp_img[0].data.cpu().mul(0.5).add(0.5))
            # put the ending samples into list
            for i in range(10):
                imgs.append(x_end[i].data)
            vutils.save_image(imgs, "pictures/interpo_{}_random.png".format(IMG_NAME), nrow=10)

    ############### 3. do the interpolation for LatMM ###############
    if mode == "Interpolation" and not hparams.Mixture.naive:
        x = batch[0]
        y = batch[1]
        digits = list(range(10))
        factor = 2
        x_ordered1 = []
        x_ordered2 = []
        for num in digits:
            assert x[y==num].size(0)>1
            x_ordered1.append(x[y==num][0])
            x_ordered2.append(x[y==num][-1])
        x_ordered = x_ordered1 + x_ordered2
        newbatch = next(iter(dataloader))
        # file x to make its batch size as 64
        for i in range(factor*len(digits), batch_size):
            x_ordered.append(newbatch[0][i])
        x_ordered = torch.stack(x_ordered)
        vutils.save_image(x_ordered.mul(0.5).add(0.5), "pictures/sample.png", nrow=10)
        sel_z = LatMMmap_encoding(graph=graph, x=x_ordered.cuda())
    
        #assert False
        x_start = x_ordered[:10]
        z_start = sel_z[:10]

        
        z_end = sel_z[10:20]
        x_end = x_ordered[10:20]
        
        points = 9
        weight = np.linspace(0.1, 0.9, points)
        z_iep = []
        counter = 0
        for i in weight:
            for j,_ in enumerate(z_start):
                z_iep.append(torch.lerp(z_start[j], z_end[j], i))
                counter +=1
        
        imgs = []
        # put the starting samples into the list
        for i in range(10):
            imgs.append(x_start[i].data)
        # do the generation 
        if True:            
            for i in range(counter):
                tmp_img = graph.get_component()(z=z_iep[i].expand(batch_size, z_iep[0].size(0), z_iep[0].size(1), z_iep[0].size(2)), reverse= True)
                imgs.append(tmp_img[0].data.cpu())
            # put the ending samples into list
            for i in range(10):
                imgs.append(x_end[i].data)
            # saving
            imgs = torch.stack(imgs)
            vutils.save_image(imgs.mul(0.5).add(0.5), "pictures/interpo_{}_sample.png".format(IMG_NAME), nrow=10)
        
