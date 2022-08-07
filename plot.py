import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str)

def plot_sparse_ratio_vs_alpha(csv_file, fig_folder):
    df = pd.read_csv(csv_file)
    plt.figure()
    for column in df.columns:
        if 'zero_rate3' in column:
            label = column.split(':')[0]
            plt.plot(df['alpha'], df[column], label=f"{label} sparse ratio")
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Sparse Ratio %')
    plt.title(r"The sparse ratio for linear regression, $w_i = 0$ if $|w_i| < 10^{-3}$")
    plt.savefig(os.path.join(fig_folder, 'sparse_ratio_1e-3.pdf'))
    plt.savefig(os.path.join(fig_folder, 'sparse_ratio_1e-3.png'))


    plt.figure()
    for column in df.columns:
        if 'zero_rate6' in column:
            label = column.split(':')[0]
            plt.plot(df['alpha'], df[column], label=f"{label} sparse ratio")
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Sparse Ratio %')
    plt.title(r"The sparse ratio for linear regression, $w_i = 0$ if $|w_i| < 10^{-6}$")
    plt.savefig(os.path.join(fig_folder, 'sparse_ratio_1e-6.pdf'))
    plt.savefig(os.path.join(fig_folder, 'sparse_ratio_1e-6.png'))

if __name__ == '__main__':
    args = parser.parse_args()

    plot_sparse_ratio_vs_alpha(
        csv_file=os.path.join(args.data_folder, 'metrics.csv'),
        fig_folder=args.data_folder)
