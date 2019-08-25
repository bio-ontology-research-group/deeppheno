from matplotlib import pyplot as plt
import click as ck
import numpy as np

@ck.command()
def main():
    groups, performance, cs, colors, yerr = get_data()
    cs = map(lambda x: 'C={}'.format(x), cs)
    n = len(groups)
    y_pos = np.arange(n)
    fig, ax = plt.subplots()
    plt.rc('text', usetex=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('none')
    ax.tick_params(axis='both', which='major', labelsize='x-large')
    ax.bar(
        y_pos, performance, align='center', color=colors,
        edgecolor="none")
    x = np.arange(-1, n, 0.2)
    y = [performance[-1]] * len(x)
    ax.plot(x, y, 's', markersize=3, color='#00539f', markeredgecolor="none")
    y = [performance[-2]] * len(x)
    ax.plot(x, y, 's', markersize=3, color='#c4302b', markeredgecolor="none")
    tcolors = ['black'] * n
    tcolors[-1] = tcolors[-2] = 'white'
    for xloc, c in zip(y_pos, cs):
        ax.text(xloc, 0.05, c, clip_on=True, rotation='vertical',
                va='bottom', ha='center', color=tcolors[xloc], size='x-large')
    plt.xticks(y_pos, groups, rotation=45, ha="right", size='x-large')
    plt.xlim([-1, n])
    plt.ylabel(r'$F_{\max}$', size='x-large')
    plt.title(r'\textbf{Human Phenotype Ontology}')
    plt.tight_layout()
    plt.savefig('hpo.eps')
    plt.show()



def get_data():
    groups = ['DeepPheno', 'EVEX', 'Rost Lab', 'Tian Lab', 'HPO2GO', 'Anacleto Lab', 'Gough Lab', 'KernelFusion', 'INGA-Tosatto', 'BAR++', 'Naive', 'BLAST']
    performance = [0.40, 0.36, 0.36, 0.32, 0.31, 0.29, 0.26, 0.22, 0.21, 0.11, 0.36, 0.22]
    cs = [1.00, 1.00, 1.00, 0.97, 0.97, 1.00, 1.00, 1.00, 0.67, 0.25, 1.00, 0.99]
    colors = ['#999999'] * len(groups)
    colors[0] = '#42aaff'
    colors[-1] = '#00539f'
    colors[-2] = '#c4302b'
    yerr = [0.02] * len(groups)
    return groups, performance, cs, colors, yerr

if __name__ == '__main__':
    main()
