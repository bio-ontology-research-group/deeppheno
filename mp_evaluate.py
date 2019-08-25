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
    '--train-data-file', '-trdf', default='data/mouse.pkl',
    help='Data file with training features')
@ck.option(
    '--test-data-file', '-tsdf', default='data/mp_predictions.pkl',
    help='Test data file')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--diamond-scores-file', '-dsf', default='data/diamond.res',
    help='Diamond output')
@ck.option(
    '--ont', '-o', default='mf',
    help='GO subontology (bp, mf, cc)')
@ck.option(
    '--alpha', '-a', default=50,
    help='Alpha for for combining scores')
def main(train_data_file, test_data_file, terms_file,
         diamond_scores_file, ont, alpha):

    alpha /= 100.0
    mp = Ontology('data/mp.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    test_df = pd.read_pickle(test_data_file)
    annotations = train_df['mp_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['mp_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    mp.calculate_ic(annotations)
    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i

    # GO2HP preds
    rules = {}
    with open('data/go2hp.txt') as f:
        for line in f:
            it = line.strip().split('\t')
            go_id = it[0].replace('_', ':')
            mp_ids = list(map(lambda x: x.replace('_', ':'), it[1:]))
            if go_id not in rules:
                rules[go_id] = []
            rules[go_id] = mp_ids
    pheno2go_preds = {}
    for i, row in enumerate(test_df.itertuples()):
        prot_id = row.proteins
        if prot_id not in pheno2go_preds:
            pheno2go_preds[prot_id] = {}
        for item in row.deepgo_annotations:
            go_id, score = item.split('|')
            if go_id in rules:
                for mp_id in rules[go_id]:
                    pheno2go_preds[prot_id][mp_id] = max(
                        float(score),
                        pheno2go_preds[prot_id].get(mp_id, 0))
    
    labels = test_annotations
    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    for t in range(101):
        threshold = t / 100.0
        preds = []
        for i, row in enumerate(test_df.itertuples()):
            prot_id = row.proteins
            annots_dict = {} #pheno2go_preds[prot_id]
            for j, score in enumerate(row.preds):
                mp_id = terms[j]
                annots_dict[mp_id] = max(score, annots_dict.get(mp_id, 0))
                
            annots = set()
            for mp_id, score in annots_dict.items():
                if score >= threshold:
                    annots.add(mp_id)
            new_annots = set()
            for mp_id in annots:
                new_annots |= mp.get_anchestors(mp_id)
            preds.append(new_annots)
        
    
        # Filter classes
        
        fscore, prec, rec, s = evaluate_annotations(mp, labels, preds)
        precisions.append(prec)
        recalls.append(rec)
        print(f'Fscore: {fscore}, S: {s}, threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
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
    plt.figure()
    lw = 2
    plt.plot(recalls, precisions, color='darkorange',
             lw=lw, label=f'AUPR curve (area = {aupr:0.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Area Under the Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.savefig(f'aupr_{ont}_{alpha:0.2f}.pdf')
    df = pd.DataFrame({'precisions': precisions, 'recalls': recalls})
    df.to_pickle(f'PR_{ont}_{alpha:0.2f}.pkl')

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
