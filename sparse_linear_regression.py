import argparse
import os
from collections import defaultdict
import pickle

import numpy as np

np.set_printoptions(precision=8, suppress=True)
import pandas as pd

from utils import eval_over_datasets
from routines import run_l1_regression, run_lasso, run_redundent_sparse_regression
from data import isotropic_predictor_data

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--predictor_dim', type=int, default=100)
parser.add_argument('--respond_dim', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--noisy_variance', type=float, default=0.1)
parser.add_argument('--optname', type=str, default='Adam')
parser.add_argument('--regularization', type=str, default='rs', choices=['lasso', 'rs', 'l1'])
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_alpha', type=int, default=20)
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

    alpha_range = np.arange(args.num_alpha) / (args.num_alpha - 1) * \
                  np.max(np.abs(non_zero_coefficient)) * 1.1

    # lasso_weights_sorted = defaultdict(list)
    # l1_weights_sorted = defaultdict(list)
    # rs_weights_sorted = defaultdict(list)

    weights = dict()
    metrics_output = defaultdict(list)
    metrics_trajectory = dict()

    for alpha in alpha_range.tolist():
        print('alpha = ', alpha)
        metrics_output['alpha'].append(alpha)

        if args.regularization == 'lasso':
            fetch = run_lasso(alpha, x, y)
            weights[f'lasso:alpha={alpha}'] = fetch['weights']
            eval_fetch = eval_over_datasets(x, y, fetch['weights'], alpha)
            for k in eval_fetch:
                metrics_output[k].append(eval_fetch[k])
            metrics_output['time'].append(fetch['time'])
            print('==> (baseline) lasso', eval_fetch)

        elif args.regularization == 'l1':
            fetch = run_l1_regression(alpha, x, y,
                                      args.optname,
                                      args.batch_size,
                                      args.lr,
                                      args.epoch,
                                      args.device)
            weights[f'l1:alpha={alpha}'] = fetch['weights']
            metrics_trajectory[f'l1:alpha={alpha}'] = fetch['metric_list']
            eval_fetch = eval_over_datasets(x, y, fetch['weights'], alpha)
            for k in eval_fetch:
                metrics_output[k].append(eval_fetch[k])
            metrics_output['time'].append(fetch['time'])
            print('==> (baseline) l1', eval_fetch)

        elif args.regularization == 'rs':
            fetch = run_redundent_sparse_regression(alpha, x, y,
                                                    args.optname,
                                                    args.batch_size,
                                                    args.lr,
                                                    args.epoch,
                                                    args.device)
            weights[f'rs:alpha={alpha}'] = fetch['weights']
            metrics_trajectory[f'rs:alpha={alpha}'] = fetch['metric_list']
            eval_fetch = eval_over_datasets(x, y, fetch['weights'], alpha)
            for k in eval_fetch:
                metrics_output[k].append(eval_fetch[k])
            metrics_output['time'].append(fetch['time'])
            print('==> (redundant) rs', eval_fetch)

    pd.DataFrame(data=metrics_output).to_csv(
        os.path.join(args.output_folder, 'metrics.csv'), index=False)
    with open(os.path.join(args.output_folder, 'weights_dict.pickle'), 'wb') as f:
        pickle.dump(weights, f)
    with open(os.path.join(args.output_folder, 'metrics_traject.pickle'), 'wb') as f:
        pickle.dump(metrics_trajectory, f)
