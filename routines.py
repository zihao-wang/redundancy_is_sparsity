import time

import torch
from tqdm import trange
from sklearn.linear_model import Lasso, LassoLars

from models import RSLinearRegression, LinearRegression
from utils import eval_over_datasets


class EarlyEscapeZeroRate:
    def __init__(self, patience, beta=0.99) -> None:
        self.patience = patience
        self.beta = beta
        self.past_zero_rate = -1

    def check_escape(self, zero_rate):
        self.past_zero_rate = self.beta * \
                              self.past_zero_rate + (1 - self.beta) * zero_rate

        if self.past_zero_rate >= zero_rate:
            return True
        else:
            return False


def run_lasso(alpha, x, y, method='default'):
    _, predictor_dim = x.shape
    _, respond_dim = y.shape

    if method == 'LARS':
        lasso_regressor = LassoLars(alpha=alpha, normalize=False, max_iter=100000)
    else:
        lasso_regressor = Lasso(alpha=alpha, max_iter=100000)

    t = time.time()
    lasso_regressor.fit(x, y)
    t = time.time() - t

    _coef = lasso_regressor.coef_
    print(_coef.shape)
    weights = _coef.T

    return {'time': t,
            'weights': weights}


def run_rs_regression(alpha, x, y,
                      optname='SGD',
                      epoch=200,
                      batch_size=512,
                      lr=1e-4,
                      device='cuda:0',
                      loss_less_than=0,
                      zero_rate_greater_than=1,
                      zero_rate_ratios=[0.5, 0.75, 0.9, 0.99, 0.999, 1, 1.01],
                      eval_every_epoch=100):
    _, predictor_dim = x.shape
    _, respond_dim = y.shape

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = RSLinearRegression(predictor_dim=predictor_dim,
                               respond_dim=respond_dim)

    # using weight decay for L2 regularization
    model.to(device)
    optimizer = getattr(torch.optim, optname)(
        model.parameters(), lr=lr, weight_decay=alpha)

    early_escape_zero_rate = EarlyEscapeZeroRate(100)

    metric_list = []

    t = time.time()
    with trange(epoch) as titer:
        for e in titer:
            metric = {}
            total_loss = 0
            for x_batch, y_batch in dataloader:
                y_pred = model(x_batch)
                l1reg = model.L1_reg()
                loss = torch.sum((y_batch - y_pred) ** 2) / 2 / y_pred.size(0)
                assert not torch.isnan(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() + alpha * l1reg.item()

            epoch_loss = total_loss / len(dataloader)
            metric['epoch_loss'] = epoch_loss
            metric['epoch'] = e + 1
            metric['time'] = time.time() - t

            if (e + 1) % eval_every_epoch == 0:
                m = eval_over_datasets(x, y, model.get_weights(), alpha)
                metric.update(m)

                should_early_stop = True

                if epoch_loss < loss_less_than:
                    metric['loss_smaller_than_threshold'] = True
                    should_early_stop = should_early_stop and True
                else:
                    metric['loss_smaller_than_threshold'] = False
                    should_early_stop = should_early_stop and False

                for zrr in zero_rate_ratios:
                    if metric['zero_rate6'] >= zero_rate_greater_than * zrr:
                        metric[f'zero_rate_greater_than_threshold:{zrr}'] = True
                        should_early_stop = should_early_stop and True
                    else:
                        metric[f'zero_rate_greater_than_threshold:{zrr}'] = False
                        should_early_stop = should_early_stop and False

                metric_list.append(metric)

                if should_early_stop:
                    break

                if early_escape_zero_rate.check_escape(metric['zero_rate12']):
                    break

            if metric_list:
                titer.set_postfix(metric_list[-1])

    t = time.time() - t

    weights = model.get_weights().reshape([predictor_dim, respond_dim])

    return {'time': t,
            'weights': weights,
            'metric_list': metric_list}


def run_l1_regression(alpha, x, y,
                      optname='SGD',
                      epoch=200,
                      batch_size=512,
                      lr=1e-4,
                      device='cuda:0',
                      loss_requirement=0,
                      eval_every_epoch=True):
    _, predictor_dim = x.shape
    _, respond_dim = y.shape

    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = LinearRegression(predictor_dim=predictor_dim,
                             respond_dim=respond_dim)

    # using L1 regularization
    model.to(device)
    optimizer = getattr(torch.optim, optname)(
        model.parameters(), lr=lr)

    early_escape_zero_rate = EarlyEscapeZeroRate(100)

    metric_list = []

    t = time.time()
    for e in range(epoch):
        metric = {}
        total_loss = 0
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            l1reg = model.L1_reg()
            loss = torch.sum((y_batch - y_pred) ** 2) / 2 / \
                   y_pred.size(0) + alpha * l1reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        metric['epoch_loss'] = epoch_loss
        metric['epoch'] = e + 1
        if eval_every_epoch:
            m = eval_over_datasets(x, y, model.get_weights(), alpha)
            metric.update(m)
            if early_escape_zero_rate.check_escape(metric['zero_rate12']):
                break

        metric_list.append(metric)

        if epoch_loss < loss_requirement:
            break

    t = time.time() - t

    weights = model.get_weights().reshape((predictor_dim, respond_dim))

    return {'time': t,
            'weights': weights,
            'metric_list': metric_list}
