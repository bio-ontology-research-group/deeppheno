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
    '--subont', '-s', default='organ',
    help='Sub Ontology')
def main(subont):
    avg = {}
    for fold in range(1,6):
        with open(f'fold{fold}_data-cafa/predictions.pkl.{subont}.res') as f:
            lines = f.read().splitlines()
            res = lines[-1].split(', ')
            for item in res:
                it = item.split(': ')
                if it[0] not in avg:
                    avg[it[0]] = 0.0
                avg[it[0]] += float(it[1])
    for key in avg:
        avg[key] /= 5

    avg_flat = {}
    for fold in range(1,6):
        with open(f'fold{fold}_data-cafa/predictions.pkl_flat.pkl.{subont}.res') as f:
            lines = f.read().splitlines()
            res = lines[-1].split(', ')
            for item in res:
                it = item.split(': ')
                if it[0] not in avg_flat:
                    avg_flat[it[0]] = 0.0
                avg_flat[it[0]] += float(it[1])
    for key in avg_flat:
        avg_flat[key] /= 5

    with open(f'data-cafa/htd.{subont}.res') as f:
        for line in f:
            it = line.strip().split('\t')
            print('\\hline')
            print(' & '.join(it) + ' \\\\')
    auc = avg_flat['AUROC']
    fmax = avg_flat['Fmax']
    p = avg_flat['Prec']
    r = avg_flat['Rec']
    print(f'DeepPhenoFlat & {auc:0.2f} & {fmax:0.2f} & {p:0.2f} & {r:0.2f} \\\\')
    print('\\hline')
    auc = avg['AUROC']
    fmax = avg['Fmax']
    p = avg['Prec']
    r = avg['Rec']
    print(f'DeepPheno & {auc:0.2f} & {fmax:0.2f} & {p:0.2f} & {r:0.2f} \\\\')
    
if __name__ == '__main__':
    main()
