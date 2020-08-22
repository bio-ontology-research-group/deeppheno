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
@ck.option(
    '--ont', '-o', default='organ',
    help='model method')
def main(method, ont):
    # res = {}
    # for fold in range(1,6):
    #     with open(f'fold{fold}_data-cafa/predictions{method}.pkl.{ont}.res') as f:
    #         lines = f.read().splitlines()
    #         items = lines[-1].split(', ')
    #         for item in items:
    #             it = item.split(': ')
    #             if it[0] not in res:
    #                 res[it[0]] = []
    #             res[it[0]].append(float(it[1]))
    #     with open(f'fold{fold}_data-cafa/predictions{method}.pkl.auc.{ont}.res') as f:
    #         lines = f.read().splitlines()
    #         auc = float(lines[-1])
    #         if 'mauc' not in res:
    #             res['mauc'] = []
    #         res['mauc'].append(auc)
            
    # avg = {}
    # avg_err = {}
    # for key in res:
    #     res[key] = np.array(res[key])
    #     avg[key] = np.mean(res[key])
    #     avg_err[key] = np.mean(np.abs(res[key] - avg[key]))
        
    # res_flat = {}
    # for fold in range(1,6):
    #     with open(f'fold{fold}_data-cafa/predictions{method}.pkl_flat.pkl.{ont}.res') as f:
    #         lines = f.read().splitlines()
    #         items = lines[-1].split(', ')
    #         for item in items:
    #             it = item.split(': ')
    #             if it[0] not in res_flat:
    #                 res_flat[it[0]] = []
    #             res_flat[it[0]].append(float(it[1]))
    #     with open(f'fold{fold}_data-cafa/predictions{method}.pkl_flat.pkl.auc.{ont}.res') as f:
    #         lines = f.read().splitlines()
    #         auc = float(lines[-1])
    #         if 'mauc' not in res_flat:
    #             res_flat['mauc'] = []
    #         res_flat['mauc'].append(auc)
    
    # avg_flat = {}
    # avg_flat_err = {}
    # for key in res_flat:
    #     res_flat[key] = np.array(res_flat[key])
    #     avg_flat[key] = np.mean(res_flat[key])
    #     avg_flat_err[key] = np.mean(np.abs(res_flat[key] - avg_flat[key]))

    # auc = avg_flat['mauc']
    # fmax = avg_flat['Fmax']
    # smin = avg_flat['Smin']
    # aupr = avg_flat['AUPR']
    # auce = avg_flat_err['mauc']
    # fmaxe = avg_flat_err['Fmax']
    # smine = avg_flat_err['Smin']
    # aupre = avg_flat_err['AUPR']
    # print(f'DeepPhenoFlat & {fmax:0.3f} $\pm$ {fmaxe:0.3f} & {smin:0.3f} $\pm$ {smine:0.3f} & {aupr:0.3f} $\pm$ {aupre:0.3f} & {auc:0.3f}  $\pm$ {auce:0.3f} \\\\')
    # print('\\hline')

    # auc = avg['mauc']
    # fmax = avg['Fmax']
    # smin = avg['Smin']
    # aupr = avg['AUPR']
    # auce = avg_err['mauc']
    # fmaxe = avg_err['Fmax']
    # smine = avg_err['Smin']
    # aupre = avg_err['AUPR']
    # print(f'DeepPheno & {fmax:0.3f} $\pm$ {fmaxe:0.3f} & {smin:0.3f} $\pm$ {smine:0.3f} & {aupr:0.3f} $\pm$ {aupre:0.3f} & {auc:0.3f} $\pm$ {auce:0.3f} \\\\')

    # res_gd = {}
    # gd = {}
    # gd_err = {}
    # for fold in range(1,6):
    #     with open(f'fold{fold}_data/sim_gene_disease{method}.txt.res') as f:
    #         lines = f.read().splitlines()
    #         res = lines[-1].split(' ')
    #         for i, item in enumerate(res):
    #             if i not in res_gd:
    #                 res_gd[i] = []
    #             res_gd[i].append(float(item))
    # for key in res_gd:
    #     res_gd[key] = np.array(res_gd[key])
    #     gd[key] = np.mean(res_gd[key])
    #     gd_err[key] = np.mean(np.abs(res_gd[key] - gd[key]))
        
    # print(f'{gd[0]:0.2f} {gd[1]:0.2f} {gd[2]:0.2f} {gd[3]:0.2f}')

    res_phenos = {}
    phenos = {}
    ph = {}
    ph_err = {}
    for fold in range(1,6):
        with open(f'fold{fold}_data/phenotype_results.tsv') as f:
            for line in f:
                it = line.strip().split('\t')
                if it[0] not in res_phenos:
                    res_phenos[it[0]] = []
                    phenos[it[0]] = it
                res_phenos[it[0]].append(float(it[2]))
    for key in res_phenos:
        res_phenos[key] = np.array(res_phenos[key])
        ph[key] = np.mean(res_phenos[key])
        ph_err[key] = np.mean(np.abs(res_phenos[key] - ph[key]))
        
    res = []
    for key, it in phenos.items():
        res.append((it[0], it[1], ph[key], ph_err[key], it[3], it[4]))
    res = sorted(res, key=lambda x: x[2], reverse=True)
    with open('data/phenotype_results.tsv', 'w') as f:
        f.write('HP\tLabel\tFmax\n')
        for it in res:
            f.write(f'{it[0]} & {it[1]} & {it[2]:.3f} $\pm$ {it[3]:.3f} \\\\ \n')

if __name__ == '__main__':
    main()
