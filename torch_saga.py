from copy import deepcopy
from random import randint
from typing import Callable, Dict
from models import LogisticRegression, SparseLogisticRegression

import torch
from torch import nn


class SAGA:
    def __init__(self, net: nn.Module, loss_func: nn.Module, reg_p, alpha) -> None:
        self.net = net
        self.loss_func = loss_func
        self.p = reg_p
        self.alpha = alpha

        self.latest_gradient_by_i: Dict[int: nn.Module] = {}
        self.mean_gradient: nn.Module = None

    def get_empty_grad(self):
        ret = deepcopy(self.net)
        for para in ret.parameters():
            para.data.fill_(0)
        return ret

    def add_grad(self, from_net: nn.Module, to_grad=None):
        if to_grad is None:
            to_grad = self.get_empty_grad()
        for fparam, tgrad in zip(from_net.parameters(), to_grad.parameters()):
            tgrad.data = fparam.grad.data.clone()
        return to_grad

    def compute_loss(self, net: nn.Module, data, target) -> torch.Tensor:
        pred = net(data)
        loss = self.loss_func(pred, target)
        for params in net.parameters():
            loss += torch.norm(params, p=self.p) * self.alpha
        return loss

    def run(self,
            X: torch.Tensor, y: torch.Tensor,
            learning_rate, num_steps,
            log_callback: Callable,
            log_every=100):
        '''
        Compute the saga algorithm
        inputs : neural net, dataset, learning rate, loss function, n_samples, iter_list, n_epoch
        goal : get a minimum
        return : update net with optimum parameters
        '''

        num_samples = X.size(1)
        net_opt = torch.optim.SGD(self.net.parameters())

        for i in range(num_samples):
            data, target = X[i], y[i]
            lossi = self.compute_loss(self.net, data, target)
            net_opt.zero_grad()
            lossi.backward()
            self.latest_gradient_by_i[i] = self.add_grad(self.net)
            self.add_grad(self.net, self.mean_gradient)

        for grad in self.mean_gradient.parameters():
            grad /= num_samples

        for i in range(num_steps):
            j = randint(a=0, b=num_samples-1)
            data, target = X[j], y[j]
            lossj = self.compute_loss(self.net, data, target)
            net_opt.zero_grad()
            lossj.backward()

            # new_grad = self.add_grad(self.net)
            old_grad = self.latest_gradient_by_i[j]

            for para, ograd, mgrad in zip(
                self.net.parameters(),
                old_grad.parameters(),
                self.mean_gradient.parameters()
            ):
                saga_update = para.grad.data - ograd.data + mgrad.data
                mgrad.data = (para.grad.data - ograd.data) / num_samples
                ograd.data = para.grad.data.clone()
                para.data.sub_(saga_update * learning_rate)

            if (i + 1) % log_every == 0:
                log_callback(self.net)
