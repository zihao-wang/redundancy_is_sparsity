import os
import pickle
import os.path as osp
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

methods = ['lasso', 'rs']
input_dims = [100, 1000, 10000, 100000]
output_dims = [1, 100]
alphas = [1, 2, 3, 4, 5]
zr_ratios = [0.75, 0.9, 0.99]
zr_markers = ['<', '^', '>', 's', 'o']


def get_rs_min_time(input_dim, output_dim, alpha, criteria):
    folder = osp.join('output/HighDimLinearRegression', f'{input_dim}_{output_dim}')
    min_time = None
    for fn in os.listdir(folder):
        if 'rs' in fn and not fn.endswith('.csv') and fn.startswith(f'{alpha}-'):
            print(fn)
            fp = osp.join(folder, fn)
            t = None
            with open(fp, 'rt') as f:
                for line in f.readlines():
                    record = json.loads(line)
                    if record[criteria]:
                        t = record['time']
                        break

    if min_time:
        min_time = min(min_time, t)
    else:
        min_time = t
    return min_time


def get_rs_min_loss(input_dim, output_dim, alpha, criteria):
    folder = osp.join('output/HighDimLinearRegression', f'{input_dim}_{output_dim}')
    min_time = None
    for fn in os.listdir(folder):
        if 'rs' in fn and not fn.endswith('.csv') and fn.startswith(f'{alpha}-'):
            print(fn)
            fp = osp.join(folder, fn)
            t = None
            with open(fp, 'rt') as f:
                for line in f.readlines():
                    record = json.loads(line)
                    if record[criteria]:
                        t = record['total']
                        break

    if min_time:
        min_time = min(min_time, t)
    else:
        min_time = t
    return min_time


def get_lasso_time(input_dim, output_dim, method, alpha):
    folder = osp.join('output/HighDimLinearRegression', f'{input_dim}_{output_dim}')
    min_time = None
    t = None
    for fn in os.listdir(folder):
        if method.lower() in fn:
            fp = osp.join(folder, fn)
            df = pd.read_csv(fp)
            record = df[df.alpha == alpha].to_dict('list')
            t = record['time'][0]
    if min_time:
        min_time = min(min_time, t)
    else:
        min_time = t
    return min_time


def get_lasso_loss(input_dim, output_dim, method, alpha):
    folder = osp.join('output/HighDimLinearRegression', f'{input_dim}_{output_dim}')
    min_loss = None
    t = None
    for fn in os.listdir(folder):
        if method.lower() in fn:
            fp = osp.join(folder, fn)
            df = pd.read_csv(fp)
            record = df[df.alpha == alpha].to_dict('list')
            t = record['total'][0]
    if min_loss:
        min_loss = min(min_loss, t)
    else:
        min_loss = t
    return min_loss


os.makedirs('output/high_dim', exist_ok=True)

for a in alphas:
    for outdim in output_dims:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax2 = ax.twinx()
        loss_lnlist = []
        loss_lblist = []
        time_lnlist = []
        time_lblist = []

        x = input_dims
        y = [get_lasso_time(indim, outdim, 'lasso', a) for indim in input_dims]
        y2 = [get_lasso_loss(indim, outdim, 'lasso', a) for indim in input_dims]
        ln1, = ax.plot(x, y, "r-")
        time_lnlist.append(ln1)
        time_lblist.append(f"Lasso time")
        # ln2, = ax2.plot(x, y2, "r:")
        # loss_lnlist.append(ln2)
        # loss_lblist.append(f"Lasso loss")

        lasso_loss = np.asarray(y2)

        x = input_dims
        y = [get_lasso_time(indim, outdim, 'lars', a) for indim in input_dims]
        y2 = [get_lasso_loss(indim, outdim, 'lars', a) for indim in input_dims]
        ln1, = ax.plot(x, y, "y-")
        time_lnlist.append(ln1)
        time_lblist.append(f"LARS time")
        # ln2, = ax2.plot(x, y2, "y:")
        # loss_lnlist.append(ln2)
        # loss_lblist.append(f"LARS loss")

        for _c, _m in zip(zr_ratios, zr_markers):
            cri = f'zero_rate_greater_than_threshold:{_c}'
            y = [get_rs_min_time(indim, outdim, a, cri) for indim in input_dims]
            y2 = [get_rs_min_loss(indim, outdim, a, cri) for indim in input_dims]
            current_loss = np.asarray(y2)
            if y[0] is None or y[1] is None:
                continue
            ln1, = ax.plot(x, y, "b-", marker=_m, alpha=0.5)
            time_lnlist.append(ln1)
            time_lblist.append(f"RS {_c} time")
            ln2, = ax2.plot(x, (current_loss - lasso_loss) / lasso_loss, "b:", marker=_m, alpha=0.5)
            loss_lnlist.append(ln2)
            loss_lblist.append(f"RS {_c} related loss difference")

        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Time')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax2.set_ylabel('Loss')
        # ax2.set_yscale('log')
        # ax2.set_ylim(1e3, 1e7)
        ax.set_title(rf'$\alpha={a / 5} \alpha_m$, Output dimension={outdim}')
        plt.legend(time_lnlist + loss_lnlist, time_lblist + loss_lblist, ncol=1, bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()
        plt.savefig(f"output/high_dim/alpha={a}_output_dim={outdim}.pdf")
        plt.savefig(f"output/high_dim/alpha={a}_output_dim={outdim}.png")
