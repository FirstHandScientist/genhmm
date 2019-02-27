import re
import os
import torch
import torch.nn.functional as F

import datetime
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from .utils import save, load, plot_prob
from .config import JsonConfig
from .models import Glow
from . import thops


class Trainer(object):
    def __init__(self, graph, graph_prior, optim, lrschedule, loaded_step,
                 devices, data_device,
                 dataset, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)
        # set members
        # append date info
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")
        self.log_dir = os.path.join(hparams.Dir.log_root, "log_" + date)
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoints_gap = hparams.Train.checkpoints_gap
        self.prior_gap = hparams.Train.prior_gap
        self.max_checkpoints = hparams.Train.max_checkpoints
        # set the mixture prior
        self.num_component = hparams.Mixture.num_component
        self.mix_prior = torch.ones(self.num_component)
        # model relative
        self.graph = graph

        self.optim = optim
        self.weight_y = hparams.Train.weight_y
        # grad operation
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm
        # copy devices from built graph
        self.devices = devices
        self.data_device = data_device
        # number of training batches
        self.batch_size = hparams.Train.batch_size
        self.data_loader = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                    #   num_workers=8,
                                      shuffle=True,
                                      drop_last=True)
        self.n_epoches = (hparams.Train.num_batches+len(self.data_loader)-1)
        self.n_epoches = self.n_epoches // len(self.data_loader)
        self.global_step = 0
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step
        # data relative
        self.y_classes = hparams.Glow.y_classes
        self.y_condition = hparams.Glow.y_condition
        self.y_criterion = hparams.Criterion.y_condition
        assert self.y_criterion in ["multi-classes", "single-class"]

        # log relative
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.plot_gaps = hparams.Train.plot_gap
        self.inference_gap = hparams.Train.inference_gap

        # mixture setting
        self.naive = hparams.Mixture.naive
        # set latent posterior
        self.num_component = hparams.Mixture.num_component
        self.graph_prior = torch.ones(hparams.Mixture.num_component)/hparams.Mixture.num_component
        self.model_prior = torch.FloatTensor(hparams.Mixture.num_component).to(self.data_device)
        # source mixture setting
        self.regulate_std = hparams.Mixture.regulate_std
        #self.regulator_std = hparams.Mixture.regulator_std
        self.warm_start = True if len(hparams.Train.warm_start)>0 else False
        
    def train(self):
        # set to training state
        if self.naive:
            for i in range(self.num_component):
                self.graph.get_component(i).train()
        else:
            self.graph.get_component().train()
        self.global_step = self.loaded_step
        # begin to train
        for epoch in range(self.n_epoches):
            try:
                print("epoch: {}, loss {}, prior {}, prior_in_graph {}".format(epoch, loss.data, self.graph_prior, self.graph.get_prior()))
            except NameError:
                print("epoch {}, loss {}, prior {}".format(epoch, "Show in next epoch", self.graph_prior))
            progress = tqdm(self.data_loader)
            for i_batch, batch in enumerate(progress):
                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])
                
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)
                # get batch data
                batch = {"x": batch[0], "y":batch[1]}
                for k in batch:
                    batch[k] = batch[k].to(self.data_device)
                
                x = batch["x"]
                y = None
                y_onehot = None
                if self.y_condition:# not entered at this stage
                    if self.y_criterion == "multi-classes":
                        assert "y_onehot" in batch, "multi-classes ask for `y_onehot` (torch.FloatTensor onehot)"
                        y_onehot = batch["y_onehot"]
                    elif self.y_criterion == "single-class":
                        assert "y" in batch, "single-class ask for `y` (torch.LongTensor indexes)"
                        y = batch["y"]
                        y_onehot = thops.onehot(y, num_classes=self.y_classes)
                                
                # at first time, initialize ActNorm
                if self.global_step == 0 and self.warm_start is False:
                    self.graph.update_prior(self.graph_prior)
                
                    self.graph(x[:self.batch_size // len(self.devices), ...],
                               y_onehot[:self.batch_size // len(self.devices), ...] if y_onehot is not None else None)
                    
                
                # forward phase and loss calculate
                #assert self.graph_prior.sum()==1, ("prior should sum to 1")
                if self.naive:
                    z, nll = self.graph(x=x, y_onehot=y_onehot)
                    nlog_joint_prob =nll - torch.log(self.graph_prior.unsqueeze(1).expand_as(nll)+1e-6).to(self.data_device)
                    with torch.no_grad():
                        tmp_sum = torch.log( torch.sum( torch.exp(-nlog_joint_prob), dim=[0]) ).to(self.data_device)                
                        nlog_gamma = nlog_joint_prob + tmp_sum.expand_as(nlog_joint_prob)
                    loss_generative =torch.sum( torch.exp(-nlog_gamma) * nlog_joint_prob) /self.batch_size
                    loss = loss_generative
                else:
                    z, gaussian_nlogp, nlogdet, reg_prior_logp = self.graph(x=x, y_onehot=y_onehot,regulate_std=self.regulate_std)
                    gaussian_nlogp = gaussian_nlogp * thops.pixels(x)
                    nlogdet = nlogdet * thops.pixels(x)

                    nlog_joint_prob = gaussian_nlogp - torch.log(self.graph_prior.unsqueeze(1)+1e-8).to(self.data_device)
                    with torch.no_grad():
                        min_nlog_joint_prob, _ = nlog_joint_prob.min(dim=0)
                        delta_nlog_joint_prob = nlog_joint_prob - min_nlog_joint_prob

                        tmp_sum = torch.log( torch.sum( torch.exp(-delta_nlog_joint_prob), dim=[0]) ).to(self.data_device)                
                        nlog_gamma = delta_nlog_joint_prob + tmp_sum.expand_as(delta_nlog_joint_prob)
                    loss_generative = (torch.mean(torch.sum( torch.exp(-nlog_gamma) * nlog_joint_prob, dim=0)) + torch.mean(nlogdet))/thops.pixels(x)
                    loss_std = 0
                    if self.regulate_std:
                        loss_std = -reg_prior_logp.mean()
                    loss = loss_generative + loss_std

                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/loss_generative", loss_generative, self.global_step)
                    if self.y_condition:
                        self.writer.add_scalar("loss/loss_classes", loss_classes, self.global_step)
      
                
                # clear buffers
                self.graph.zero_grad()
                self.optim.zero_grad()

                # backward
                loss.backward()
                
                # operate grad
                
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                    
                
                # step
                self.optim.step()
                
                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         graph_prior=self.graph_prior,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)

                # global step
                self.global_step += 1
                # accumulate the posterior of model
                with torch.no_grad():
                    self.model_prior +=(torch.exp(-nlog_gamma.data)).mean(dim=1)
                
            # #update the prior of model
            if epoch>0 and epoch%self.prior_gap == 0:
                with torch.no_grad():
                    self.model_prior = self.model_prior/torch.sum(self.model_prior)
                    
                    self.graph_prior = self.model_prior.cpu().data
                    self.graph.update_prior(self.graph_prior)
                    print("Update prior: {}".format(self.graph_prior))
            self.model_prior.zero_()
            
        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
