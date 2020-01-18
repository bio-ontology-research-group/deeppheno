#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import sys
from collections import deque
import time
import logging
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from scipy import sparse
import math
from utils import FUNC_DICT, Ontology, NAMESPACES
from matplotlib import pyplot as plt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--method', '-m', default='',
    help='model method')
def main(method):
    # avg = {}
    # for fold in range(1,6):
    #     with open(f'fold{fold}_data/predictions{method}.pkl.res') as f:
    #         lines = f.read().splitlines()
    #         res = lines[-1].split(', ')
    #         for item in res:
    #             it = item.split(': ')
    #             if it[0] not in avg:
    #                 avg[it[0]] = 0.0
    #             avg[it[0]] += float(it[1])
    # for key in avg:
    #     avg[key] /= 5

    # avg_flat = {}
    # for fold in range(1,6):
    #     with open(f'fold{fold}_data/predictions{method}.pkl_flat.pkl.res') as f:
    #         lines = f.read().splitlines()
    #         res = lines[-1].split(', ')
    #         for item in res:
    #             it = item.split(': ')
    #             if it[0] not in avg_flat:
    #                 avg_flat[it[0]] = 0.0
    #             avg_flat[it[0]] += float(it[1])
    # for key in avg_flat:
    #     avg_flat[key] /= 5

    # auc = avg_flat['AUROC']
    # fmax = avg_flat['Fmax']
    # smin = avg_flat['Smin']
    # aupr = avg_flat['AUPR']
    # print(f'DeepPhenoFlat & {fmax:0.3f} & {smin:0.3f} & {aupr:0.3f} & {auc:0.3f} \\\\')
    # print('\\hline')

    # auc = avg['AUROC']
    # fmax = avg['Fmax']
    # smin = avg['Smin']
    # aupr = avg['AUPR']
    # print(f'DeepPheno &  {fmax:0.3f} & {smin:0.3f} & {aupr:0.3f} & {auc:0.3f} \\\\')

    gd = {}
    for fold in range(1,6):
        with open(f'fold{fold}_data/sim_gene_disease{method}.txt.res') as f:
            lines = f.read().splitlines()
            res = lines[-1].split(' ')
            for i, item in enumerate(res):
                if i not in gd:
                    gd[i] = 0.0
                gd[i] += float(item)
    for key in gd:
        gd[key] /= 5
    print(f'{gd[0]:0.2f} {gd[1]:0.2f} {gd[2]:0.2f} {gd[3]:0.2f}')
    
if __name__ == '__main__':
    main()
