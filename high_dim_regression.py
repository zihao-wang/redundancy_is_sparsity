import argparse
import os
from collections import defaultdict

import numpy as np
np.set_printoptions(precision=8, suppress=True)
import pandas as pd

from utils import eval_over_datasets
from routines import run_lasso, run_rs_regression
from data import isotropic_predictor_data


parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--predictor_dim', type=int, default=100)
parser.add_argument('--respond_dim', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--noisy_variance', type=float, default=0.1)
parser.add_argument('--optname', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_alpha', type=int, default=5)
parser.add_argument('--output_folder', type=str, default='output')

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    (x, y), trans = isotropic_predictor_data(args.num_samples,
                                             args.predictor_dim,
                                             args.respond_dim,
                                             args.noisy_variance)

    # manually set the alpha thresholding
    # adding a small value to ensure the soft threshold
    non_zero_coefficient = trans[np.nonzero(trans)] + 1e-3
    alpha_range = np.abs(non_zero_coefficient).ravel().tolist()
    alpha_range.sort()

    alpha_range = (np.arange(args.num_alpha) + 1) / args.num_alpha \
                  * np.max(np.abs(non_zero_coefficient)) * 1.01

    data = defaultdict(list)

    for alpha in alpha_range.tolist():
        print('alpha = ', alpha)
        data['alpha'].append(alpha)

        lasso_fetch = run_lasso(alpha, x, y)
        lasso_eval_fetch = eval_over_datasets(x, y, lasso_fetch['weights'], alpha)
        for k in lasso_eval_fetch:
            data[f'lasso:{k}'].append(lasso_eval_fetch[k])
        data['lasso:time'].append(lasso_fetch['time'])

        rs_fetch = run_rs_regression(alpha, x, y,
                                  args.optname,
                                  args.epoch,
                                  args.batch_size,
                                  args.lr,
                                  args.device,
                                  loss_requirement=lasso_eval_fetch['total'],
                                  eval_every_epoch=False)
        rs_eval_fetch = eval_over_datasets(x, y, rs_fetch['weights'], alpha)
        for k in rs_eval_fetch:
            data[f'rs:{k}'].append(rs_eval_fetch[k])
        data['rs:time'].append(lasso_fetch['time'])

    pd.DataFrame(data=data).to_csv(
        os.path.join(args.output_folder, 'metrics.csv'), index=False)