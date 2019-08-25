import pandas as pd
import numpy as np
import click as ck
from matplotlib import pyplot as plt


@ck.command()
def main():

    data = load_data()
    # Plot all ROC curves
    plt.figure()

    colors = ['blue', 'darkorange', 'green', 'purple', 'red', ]
    for i, color in enumerate(colors):
        plt.plot(data[i][1], data[i][2], color=color, lw=2,
                 label=f'{data[i][0]} (area = {data[i][3]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Gene-Disease association prediction')
    plt.legend(loc="lower right")
    plt.show()

def load_data():
    methods = (
        ('_naive', 'Naive'),
        ('_exp', 'DeepPhenoGO'),
        ('_iea', 'DeepPhenoIEA'),
        ('', 'DeepPheno'),
        ('_real', 'RealHPO'),
    )
    data = []
    for m, method in methods:
        df = pd.read_pickle(f'data/gene_annotations{m}.tab_auc.pkl')
        n_prots = df['auc_x'].values[-1]
        x = df['auc_x'].values / n_prots
        y = df['auc_y'].values
        auc = np.trapz(y, x)
        data.append((method, x, y, auc,))
    return data

if __name__ == '__main__':
    main()
