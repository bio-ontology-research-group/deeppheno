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
    '--benchmark-file', '-bf', default='data/benchmark/groundtruth/leafonly_HPO.txt',
    help='CAFA benchmark annotations')
@ck.option(
    '--predictions-file', '-pf', default='data/predictions.txt',
    help='Predictions file')
@ck.option(
    '--train-data-file', '-trdf', default='data/human.pkl',
    help='Data file with training features')
@ck.option(
    '--hpo-file', '-hf', default='data/hp.obo',
    help='Data file with sequences and complete set of annotations')
def main(benchmark_file, predictions_file, train_data_file, hpo_file):

    hp = Ontology(hpo_file, with_rels=True)
    noknowledge_prots = set()
    with open('data/noknowledge_targets.txt') as f:
        for line in f:
            noknowledge_prots.add(line.strip())
    
    bench_annots = {}
    with open(benchmark_file) as f:
        for line in f:
            it = line.strip().split('\t')
            t_id = it[0]
            if t_id not in noknowledge_prots:
                continue
            hp_id = it[1]
            if t_id not in bench_annots:
                bench_annots[t_id] = set()
            bench_annots[t_id] |= hp.get_anchestors(hp_id)

    pred_annots = {}
    with open(predictions_file) as f:
        for line in f:
            it = line.strip().split('\t')
            t_id = it[0]
            hp_id = it[1]
            score = float(it[2])
            if t_id not in bench_annots:
                continue
            if t_id not in pred_annots:
                pred_annots[t_id] = {}
            pred_annots[t_id][hp_id] = score
    
    train_df = pd.read_pickle(train_data_file)
    annotations = train_df['hp_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    hp.calculate_ic(annotations)
    
    labels = []
    for t_id, hps in bench_annots.items():
        labels.append(hps)
    # labels = list(map(lambda x: set(filter(lambda y: y in hp_set_anch, x)), labels))
    
    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    max_preds = None
    for t in range(0, 101):
        threshold = t / 100.0
        preds = []
        for t_id, hps in bench_annots.items():
            annots_dict = {} #pheno2go_preds[gene_id].copy()
            
            if t_id in pred_annots:
                annots_dict = pred_annots[t_id].copy()
            
            annots = set()
            for hp_id, score in annots_dict.items():
                if score >= threshold:
                    annots.add(hp_id)
            new_annots = set()
            for hp_id in annots:
                new_annots |= hp.get_anchestors(hp_id)
            preds.append(new_annots)
        
    
        # Filter classes
        
        fscore, prec, rec, s = evaluate_annotations(hp, labels, preds)
        precisions.append(prec)
        recalls.append(rec)
        print(f'Fscore: {fscore}, S: {s}, threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
            max_preds = preds
        if smin > s:
            smin = s
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')
    # plt.figure()
    # lw = 2
    # plt.plot(recalls, precisions, color='darkorange',
    #          lw=lw, label=f'AUPR curve (area = {aupr:0.2f})')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Area Under the Precision-Recall curve')
    # plt.legend(loc="lower right")
    # df = pd.DataFrame({'precisions': precisions, 'recalls': recalls})
    # df.to_pickle(f'PR.pkl')

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(labels, preds):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s


if __name__ == '__main__':
    main()
