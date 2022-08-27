import os
import argparse
import pickle
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--plot_folder", type=str, default='output/plots/with_alpha')

marker_dict = {'RS': 's', 'L1': '+'}
color_dict = {'zero_rate6': 'tab:blue',
              'zero_rate12': 'tab:orange', }


def plot_sparse_ratio_vs_alpha(csv_file_dict, key_dict, fig_folder):
    format_dict = {
        'lasso': {
            'color': 'tab:gray',
            'linestyle': 'dashed',
            'alpha': 0.8
        },
        'rs:zero_rate6': {
            'color': 'tab:blue',
            'alpha': 0.9,
            'marker': '+'
        },
        'rs:zero_rate12': {
            'color': 'tab:blue',
            'alpha': 0.5,
            'marker': '+'
        },
        'l1:zero_rate6': {
            'color': 'tab:orange',
            'alpha': 0.9,
            'marker': 'x'
        },
        'l1:zero_rate12': {
            'color': 'tab:orange',
            'alpha': 0.5,
            'marker': 'x'
        },
    }

    df_dict = {k: pd.read_csv(csv)
               for k, csv in csv_file_dict.items()}

    plt.figure(figsize=(3, 2.5))
    for label in df_dict:
        df = df_dict[label]
        prefix = label.split(':')[0]
        for key in key_dict:
            for column in df.columns:
                if key in column:
                    if 'Lasso' in label:
                        if '12' in key:
                            plt.plot(df['alpha'], df[column], label='Lasso',
                                     **format_dict['lasso'])
                    else:
                        plt.scatter(df['alpha'], df[column],
                                    label=f'{prefix}({key_dict[key]})',
                                    **format_dict[f'{prefix}:{key}'.lower()])

    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Sparse Ratio')
    plt.title(r"The sparse ratio with $\alpha$")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder, 'sparse_ratio_with_alpha.pdf'))
    plt.savefig(os.path.join(fig_folder, 'sparse_ratio_with_alpha.png'))
    plt.close()


def plot_training_trajectory(pickle_file_dict, fig_folder):
    format_dict = {
        'rs:zero_rate6': {
            'color': 'tab:blue',
            'alpha': 0.9,
        },
        'rs:zero_rate12': {
            'color': 'tab:blue',
            'alpha': 0.5,
        },
        'rs:l1': {
            'color': 'tab:cyan',
            'alpha': 0.5,
        },
        'l1:zero_rate6': {
            'color': 'tab:orange',
            'alpha': 0.9,
        },
        'l1:zero_rate12': {
            'color': 'tab:orange',
            'alpha': 0.5,
        },
        'l1:l1': {
            'color': 'tab:brown',
            'alpha': 0.5,
        },
    }
    metric_trajectory_dict = {}
    for k in pickle_file_dict:
        with open(pickle_file_dict[k], 'rb') as f:
            metric_trajectory_dict[k] = pickle.load(f)

    # get common alphas to compare
    for k in metric_trajectory_dict:
        metric_trajectory = metric_trajectory_dict[k]
        alpha = list(set(float(k.split('=')[-1])
                         for k in metric_trajectory.keys()))
        break

    for a in tqdm(alpha):
        fig, axl = plt.subplots(figsize=(6, 4))
        axr = axl.twinx()
        lines = []
        for label in metric_trajectory_dict:
            prefix = label.split(':')[0]
            metric_trajectory = metric_trajectory_dict[label]
            for k in metric_trajectory.keys():
                if a == float(k.split('=')[-1]):
                    metric_list = metric_trajectory[k]
                    break
            epoch = [m['epoch'] for m in metric_list]
            l1_loss = [m['l1'] for m in metric_list]
            zero6 = [m['zero_rate6'] for m in metric_list]
            zero12 = [m['zero_rate12'] for m in metric_list]

            lines.extend(
                axl.plot(epoch, zero6,
                         label=prefix + r" Sparse Ratio ($|w|<10^{-6}$)",
                         **format_dict[f'{prefix}:zero_rate6'.lower()]))
            lines.extend(
                axl.plot(epoch, zero12,
                         label=prefix + r" Sparse Ratio ($|w|<10^{-12}$)",
                         **format_dict[f'{prefix}:zero_rate12'.lower()]
                         ))
            # axr.plot(epoch, mse_loss, linestyle="dashed", label=f"{prefix}:MSE")
            lines.extend(
                axr.plot(epoch, l1_loss,
                         label=f"{prefix}" + r" $|W|_1$",
                         **format_dict[f'{prefix}:l1'.lower()]))
        axl.set_ylabel("Sparse Ratio")
        axl.set_ylim([-0.05, 1.05])
        axl.set_xlabel("Step")
        axl.set_title(r"Sparse ratio and $L1$ during training")
        axr.set_yscale('log')
        axl.set_xscale('log')
        axr.set_ylabel("Loss")

        labels = [l.get_label() for l in lines]
        axl.legend(lines, labels, ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_folder, f"alpha={a:.4f}.pdf"))
        fig.savefig(os.path.join(fig_folder, f"alpha={a:.4f}.png"))
        plt.close()


if __name__ == '__main__':
    args = parser.parse_args()

    csv_file_dict = {
        'Lasso': 'output/LinearRegression100_1/lasso/metrics.csv',
        # 'L1:Adam_lr_1e-1': 'output/LinearRegression100_1/l1_Adam_lr=1e-1/metrics.csv',
        # 'L1:Adam_lr_1e-2': 'output/LinearRegression100_1/l1_Adam_lr=1e-2/metrics.csv',
        # 'L1:Adam_lr_1e-3': 'output/LinearRegression100_1/l1_Adam_lr=1e-3/metrics.csv',
        # 'RS:Adam_lr_1e-1': 'output/LinearRegression100_1/rs_Adam_lr=1e-1/metrics.csv',
        # 'RS:Adam_lr_1e-2': 'output/LinearRegression100_1/rs_Adam_lr=1e-2/metrics.csv',
        # 'RS:Adam_lr_1e-3': 'output/LinearRegression100_1/rs_Adam_lr=1e-3/metrics.csv',
        'L1:SGD_lr_1e-1': 'output/LinearRegression100_1/l1_SGD_lr=1e-1/metrics.csv',
        # 'L1:SGD_lr_1e-2': 'output/LinearRegression100_1/l1_SGD_lr=1e-2/metrics.csv',
        # 'L1:SGD_lr_1e-3': 'output/LinearRegression100_1/l1_SGD_lr=1e-3/metrics.csv',
        'RS:SGD_lr_1e-1': 'output/LinearRegression100_1/rs_SGD_lr=1e-1/metrics.csv',
        # 'RS:SGD_lr_1e-2': 'output/LinearRegression100_1/rs_SGD_lr=1e-2/metrics.csv',
        # 'RS:SGD_lr_1e-3': 'output/LinearRegression100_1/rs_SGD_lr=1e-3/metrics.csv',
    }

    pickle_file_dict = {
        # 'L1:Adam_lr_1e-1': 'output/LinearRegression100_1/l1_Adam_lr=1e-1/metrics_traject.pickle',
        # 'L1:Adam_lr_1e-2': 'output/LinearRegression100_1/l1_Adam_lr=1e-2/metrics_traject.pickle',
        # 'L1:Adam_lr_1e-3': 'output/LinearRegression100_1/l1_Adam_lr=1e-3/metrics_traject.pickle',
        # 'RS:Adam_lr_1e-1': 'output/LinearRegression100_1/rs_Adam_lr=1e-1/metrics_traject.pickle',
        # 'RS:Adam_lr_1e-2': 'output/LinearRegression100_1/rs_Adam_lr=1e-2/metrics_traject.pickle',
        # 'RS:Adam_lr_1e-3': 'output/LinearRegression100_1/rs_Adam_lr=1e-3/metrics_traject.pickle',
        # 'L1:SGD_lr_1e-1': 'output/LinearRegression100_1/l1_SGD_lr=1e-1/metrics_traject.pickle',
        # 'L1:SGD_lr_1e-2': 'output/LinearRegression100_1/l1_SGD_lr=1e-2/metrics_traject.pickle',
        'L1:SGD_lr_1e-3': 'output/LinearRegression100_1/l1_SGD_lr=1e-3/metrics_traject.pickle',
        'RS:SGD_lr_1e-1': 'output/LinearRegression100_1/rs_SGD_lr=1e-1/metrics_traject.pickle',
        # 'RS:SGD_lr_1e-2': 'output/LinearRegression100_1/rs_SGD_lr=1e-2/metrics_traject.pickle',
        # 'RS:SGD_lr_1e-3': 'output/LinearRegression100_1/rs_SGD_lr=1e-3/metrics_traject.pickle',
    }

    key_dict = {
        # 'zero_rate3':    r'Ratio of $|w| < 10^{-3}}$',
        'zero_rate6': r'$|w| < 10^{-6}}$',
        # 'zero_rate9':    r'Ratio of $|w| < 10^{-9}}$',
        'zero_rate12': r'$|w| < 10^{-12}}$'
    }

    os.makedirs(args.plot_folder, exist_ok=True)

    plot_sparse_ratio_vs_alpha(
        csv_file_dict=csv_file_dict,
        key_dict=key_dict,
        fig_folder=args.plot_folder)
    plot_training_trajectory(
        pickle_file_dict=pickle_file_dict,
        fig_folder=args.plot_folder)
