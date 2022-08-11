import os
import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str)


def plot_sparse_ratio_vs_alpha(csv_file, fig_folder):
    df = pd.read_csv(csv_file)
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(14, 3))
    for ax, t in zip(axes, [3, 6, 9, 12]):
        for column in df.columns:
            if f'zero_rate{t}' in column:
                label = column.split(':')[0]
                ax.plot(df['alpha'], df[column], label=f"{label}")
        ax.legend()
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('Sparse Ratio')
        ax.set_title(rf"The sparse ratio $w_i = 0$ if $|w_i| < 1e{-t}$")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_folder, 'sparse_ratio.pdf'))
    fig.savefig(os.path.join(fig_folder, 'sparse_ratio.png'))


def plot_sparse_ratio_vs_training(pickle_file, fig_folder):
    with open(pickle_file, 'rb') as f:
        metric_trajectory = pickle.load(f)

    # get all alpha values:
    alpha = list(set(float(k.split('=')[-1])
                 for k in metric_trajectory.keys()))
    for a in alpha:
        plt.figure()
        target_keys = [f"l1:alpha={a}", f"rs:alpha={a}"]
        colors = [
            'tab:blue',
            'tab:orange',
            'tab:green',
            'tab:red',
            'tab:purple',
            'tab:brown',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'tab:cyan'
        ]
        for key, c in zip(target_keys, colors):
            metric_list = metric_trajectory[key]
            epoch = [m['epoch'] for m in metric_list]
            l1_loss = [m['l1'] for m in metric_list]
            zero3 = [m['zero_rate3'] for m in metric_list]
            zero6 = [m['zero_rate6'] for m in metric_list]
            zero9 = [m['zero_rate9'] for m in metric_list]
            zero12 = [m['zero_rate12'] for m in metric_list]

            plt.plot(epoch, zero3, label=f'{key[:2]}:1e-3', alpha=1, color=c)
            # plt.plot(epoch, zero6, label=f'{key[:2]}:1e-6', alpha=.7, color=c)
            plt.plot(epoch, zero9, label=f'{key[:2]}:1e-9', alpha=.3, color=c)
            # plt.plot(epoch, zero12, label=f'{key[:2]}:1e-12', alpha=.1, color=c)

        plt.legend()
        plt.title(fr"Zero ratio v.s. training steps under different thresholds $\alpha$ = {a:.4f}")
        plt.xlabel("Step")
        plt.ylabel("Zero Ratio")
        # plt.yscale('log')

        plt.savefig(os.path.join(fig_folder, f"alpha={a:.4f}.pdf"))
        plt.savefig(os.path.join(fig_folder, f"alpha={a:.4f}.png"))


if __name__ == '__main__':
    args = parser.parse_args()

    plot_sparse_ratio_vs_alpha(
        csv_file=os.path.join(args.data_folder, 'metrics.csv'),
        fig_folder=args.data_folder)
    plot_sparse_ratio_vs_training(
        pickle_file=os.path.join(args.data_folder, 'metrics_traject.pickle'),
        fig_folder=args.data_folder)
