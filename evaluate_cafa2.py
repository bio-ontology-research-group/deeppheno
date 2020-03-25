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
    '--train-data-file', '-trdf', default='data-cafa/human.pkl',
    help='Data file with training features')
@ck.option(
    '--test-data-file', '-tsdf', default='data-cafa/predictions.pkl',
    help='Test data file')
@ck.option(
    '--terms-file', '-tf', default='data-cafa/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--out-file', '-of', default='data-cafa/predictions_max.pkl',
    help='Results file with best Fmax predictions')
@ck.option(
    '--root-class', '-rc', default='HP:0000118',
    help='Root class for evaluation')
@ck.option(
    '--fold', '-f', default=1,
    help='Root class for evaluation')
def main(train_data_file, test_data_file, terms_file, out_file, root_class, fold):
    # Cross validation evaluation
    out_file = f'fold{fold}_' + out_file
    test_data_file = f'fold{fold}_' + test_data_file
    
    hp = Ontology('data-cafa/hp.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    test_df = pd.read_pickle(test_data_file)
    annotations = train_df['hp_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['hp_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    hp.calculate_ic(annotations)

    hp_set = set(terms)
    all_classes = hp.get_term_set(root_class)
    hp_set = hp_set.intersection(all_classes)
    hp_set.discard(root_class)
    print(len(hp_set))
    
    labels = test_annotations
    labels = list(map(lambda x: set(filter(lambda y: y in hp_set, x)), labels))

    # Compute AUC
    auc_terms = list(hp_set)
    auc_terms_dict = {v: i for i, v in enumerate(auc_terms)}
    auc_preds = np.zeros((len(test_df), len(hp_set)), dtype=np.float32)
    auc_labels = np.zeros((len(test_df), len(hp_set)), dtype=np.int32)
    for i, row in enumerate(test_df.itertuples()):
        for j, hp_id in enumerate(auc_terms):
            auc_preds[i, j] = row.preds[terms_dict[hp_id]]
            if hp_id in labels[i]:
                auc_labels[i, j] = 1
    # Compute macro AUROC
    roc_auc = 0.0
    total = 0
    for i, hp_id in enumerate(auc_terms):
        if np.sum(auc_labels[:, i]) == 0:
            continue
        total += 1
        auc = compute_roc(auc_labels[:, i], auc_preds[:, i])
        if not math.isnan(auc): 
            roc_auc += auc
        else:
            roc_auc += 1
    roc_auc /= total
    print(roc_auc)
    return

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
        preds = []
        for i, row in enumerate(test_df.itertuples()):
            gene_id = row.proteins
            annots_dict = {} 
            
            for j, score in enumerate(row.preds):
                hp_id = terms[j]
                # score = score * (1 - alpha)
                if hp_id in annots_dict:
                    annots_dict[hp_id] += score
                else:
                    annots_dict[hp_id] = score
                
            annots = set()
            for hp_id, score in annots_dict.items():
                if score >= threshold:
                    annots.add(hp_id)
            new_annots = set()
            for hp_id in annots:
                new_annots |= hp.get_anchestors(hp_id)
            new_annots = new_annots.intersection(hp_set)
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
            pmax = prec
            rmax = rec
        if smin > s:
            smin = s
    test_df['hp_preds'] = max_preds
    test_df.to_pickle(out_file)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUROC: {roc_auc:0.3f}, AUPR: {aupr:0.3f}, Fmax: {fmax:0.3f}, Prec: {pmax:0.3f}, Rec: {rmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
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
    df = pd.DataFrame({'precisions': precisions, 'recalls': recalls})
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
