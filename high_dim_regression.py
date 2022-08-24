import argparse
import os
from collections import defaultdict
import json

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



if __name__ == "__main__":
    args = parser.parse_args()
    output_folder = os.path.join("output/HighDimLinearRegression",
                                 f"{args.predictor_dim}_{args.respond_dim}")
    os.makedirs(output_folder, exist_ok=True)

    (x, y), trans = isotropic_predictor_data(args.num_samples,
                                             args.predictor_dim,
                                             args.respond_dim,
                                             args.noisy_variance,
                                             seed=666)

    # manually set the alpha thresholding
    alpha_range = np.arange(args.num_alpha) + 1

    # run lasso if there is no lasso record
    lasso_file = os.path.join(output_folder, 'lars_metrics.csv')
    # re run the lasso
    # if os.path.exists(lasso_file):
        # print(f"lasso file {lasso_file} already found")
        # lasso_df = pd.read_csv(lasso_file)
    # else:
    print(f"lasso file {lasso_file} not found")
    data = defaultdict(list)
    for alpha in alpha_range.tolist():
        data['alpha'].append(alpha)

        lasso_fetch = run_lasso(alpha, x, y, method='LARS')
        data['time'].append(lasso_fetch['time'])
        lasso_eval_fetch = eval_over_datasets(x, y, lasso_fetch['weights'], alpha)
        for k in lasso_eval_fetch:
            data[k].append(lasso_eval_fetch[k])
    lasso_df = pd.DataFrame(data)
    lasso_df.to_csv(lasso_file, index=False)
    print(lasso_df.to_string())

    lasso_file = os.path.join(output_folder, 'lasso_metrics.csv')
    lasso_df = pd.read_csv(lasso_file)

    # run rs
    data = defaultdict(list)
    for alpha in alpha_range.tolist():
        data['alpha'].append(alpha)
        lasso_record = lasso_df[lasso_df.alpha == alpha].to_dict('list')
        target_loss = lasso_record['total'][0]
        target_zero_rate = lasso_record['zero_rate12'][0]
        print(lasso_record)
        rs_fetch = run_rs_regression(alpha, x, y,
                                     args.optname,
                                     args.epoch,
                                     args.batch_size,
                                     args.lr,
                                     args.device,
                                     loss_less_than=target_loss,
                                     zero_rate_greater_than=target_zero_rate,
                                     zero_rate_ratios=[0.5, 0.75, 0.9, 0.99, 0.999, 1],
                                     eval_every_epoch=100)

        filename = os.path.join(
            output_folder,
            f'{alpha}-rs_metrics_optname_{args.optname}_lr_{args.lr}')

        with open(filename, 'wt') as f:
            for metric in rs_fetch['metric_list']:
                f.write(json.dumps(metric) + '\n')
