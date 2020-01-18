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
    '--benchmark-file', '-bf', default='data-cafa/benchmark/groundtruth/leafonly_HPO.txt',
    help='CAFA benchmark annotations')
@ck.option(
    '--train-data-file', '-trdf', default='data-cafa/human.pkl',
    help='Data file with training features')
@ck.option(
    '--hpo-file', '-hf', default='data-cafa/hp.obo',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--terms-file', '-tf', default='data-cafa/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--root-class', '-rc', default='HP:0000001',
    help='Root class for evaluation')
def main(benchmark_file, train_data_file, hpo_file, terms_file, root_class):

    hp = Ontology(hpo_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    noknowledge_prots = set()
    with open('data-cafa/noknowledge_targets.txt') as f:
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


    train_df = pd.read_pickle(train_data_file)
    naive_annots = {}
    for i, row in train_df.iterrows():
        for hp_id in row.hp_annotations:
            if hp_id in naive_annots:
                naive_annots[hp_id] += 1
            else:
                naive_annots[hp_id] = 1
    for hp_id in naive_annots:
        naive_annots[hp_id] /= 1.0 * len(train_df)
        
    annotations = train_df['hp_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    hp.calculate_ic(annotations)

    hp_set = set(terms)
    all_classes = hp.get_term_set(root_class)
    hp_set = hp_set.intersection(all_classes)
    hp_set.discard(root_class)
    print(len(hp_set))
    
    labels = []
    for t_id, hps in bench_annots.items():
        labels.append(hps)
    labels = list(map(lambda x: set(filter(lambda y: y in hp_set, x)), labels))

    # Compute AUC
    auc_terms = list(hp_set)
    auc_terms_dict = {v: i for i, v in enumerate(auc_terms)}
    auc_preds = np.zeros((len(bench_annots), len(hp_set)), dtype=np.float32)
    auc_labels = np.zeros((len(bench_annots), len(hp_set)), dtype=np.int32)
    for i in range(len(labels)):
        for j, hp_id in enumerate(auc_terms):
            auc_preds[i, j] = naive_annots[hp_id]
            if hp_id in labels[i]:
                auc_labels[i, j] = 1
    roc_auc = compute_roc(auc_labels, auc_preds)
    print(roc_auc)
    
    fmax = 0.0
    tmax = 0.0
    pmax = 0.0
    rmax = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    max_preds = None
    for t in range(0, 101):
        threshold = t / 100.0
        annots = set()
        for hp_id, score in naive_annots.items():
            if score >= threshold:
                annots.add(hp_id)
        new_annots = set()
        for hp_id in annots:
            new_annots |= hp.get_anchestors(hp_id)
        preds = []
        for t_id, hps in bench_annots.items():
            preds.append(new_annots)
        
        fscore, prec, rec, s = evaluate_annotations(hp, labels, preds)
        precisions.append(prec)
        recalls.append(rec)
        print(f'Fscore: {fscore}, S: {s}, threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            pmax = prec
            rmax = rec
            tmax = threshold
            max_preds = preds
        if smin > s:
            smin = s
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUROC: {roc_auc:0.3f}, AUPR: {aupr:0.3f}, Fmax: {fmax:0.3f}, Prec: {pmax:0.3f}, Rec: {rmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
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
