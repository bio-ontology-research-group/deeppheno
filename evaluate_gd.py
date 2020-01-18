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
from scipy.stats import rankdata

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--gene-annots-file', '-gaf', default='data/gene_annotations.tab',
    help='Data file with gene annotations')
@ck.option(
    '--dis-annots-file', '-daf', default='data/dis_annotations.tab',
    help='Disease annotations file')
@ck.option(
    '--sim-file', '-sf', default='data/sim_gene_disease.txt',
    help='Data file with similarity values')
@ck.option(
    '--gene-dis-assoc-file', '-gdaf', default='data/morbidmap.txt',
    help='Gene Disease association file')
@ck.option(
    '--fold', '-f', default=1,
    help='Fold index')
def main(gene_annots_file, dis_annots_file, sim_file, gene_dis_assoc_file, fold):
    # Cross validation evaluation
    sim_file = f'fold{fold}_' + sim_file
    gene_annots_file = f'fold{fold}_' + gene_annots_file

    genes = []
    genes_dict = {}
    with open(gene_annots_file) as f:
        for line in f:
            it = line.strip().split('\t')
            genes.append(it[0])
            genes_dict[it[0]] = len(genes_dict)
            
    diseases = []
    diseases_dict = {}
    with open(dis_annots_file) as f:
        for line in f:
            it = line.strip().split('\t')
            diseases.append(it[0])
            diseases_dict[it[0]] = len(diseases_dict)
            
    sim = np.zeros((len(genes), len(diseases)), dtype=np.float32)
    with open(sim_file) as f:
        for i in range(len(genes)):
            for j in range(len(diseases)):
                sim[i, j] = float(next(f).strip())
    
    assoc = np.zeros((len(genes), len(diseases)), dtype=np.float32)
    test_data = set()
    symb2id = {}
    with open('data/genes_to_phenotype.txt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            it = line.strip().split('\t')
            symb2id[it[1]] = it[0]
    
    with open(gene_dis_assoc_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            it = line.strip().split('\t')
            dis_id = 'OMIM:' + it[0].split(', ')[-1].split()[0]
            gene_symbols = it[1].split(', ')
            for symb in gene_symbols:
                if symb not in symb2id:
                    continue
                gene_id = symb2id[symb]
                if gene_id in genes_dict and dis_id in diseases_dict:
                    assoc[genes_dict[gene_id], diseases_dict[dis_id]] = 1
                    test_data.add((genes_dict[gene_id], diseases_dict[dis_id]))
    s = np.sum(assoc, axis=0)
    filtered = []
    dis_map = {}
    for i in range(len(s)):
        if s[i] > 0:
            dis_map[diseases_dict[diseases[i]]] = len(filtered)
            filtered.append(i)
            
    print(len(filtered))
    assoc = assoc[:, filtered]
    sim = sim[:, filtered]
    print(assoc.shape, np.sum(assoc))
    roc_auc = compute_roc(assoc, sim)
    print(roc_auc)
    
    top10 = 0
    top100 = 0
    mean_rank = 0
    ftop10 = 0
    ftop100 = 0
    fmean_rank = 0
    n = len(test_data)
    ranks = {}
    franks = {}
    with ck.progressbar(test_data) as prog_data:
        for c, d in prog_data:
            index = rankdata(-sim[c, :], method='average')
            d = dis_map[d]
            rank = index[d]
            if rank <= 10:
                top10 += 1
            if rank <= 100:
                top100 += 1
            mean_rank += rank
            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1

            # Filtered rank
            f = 1 - assoc[c, :]
            f[d] = 1
            fil = sim[c, :] * f
            index = rankdata(-fil, method='average')
            rank = index[d]
            if rank <= 10:
                ftop10 += 1
            if rank <= 100:
                ftop100 += 1
            fmean_rank += rank
            if rank not in franks:
                franks[rank] = 0
            franks[rank] += 1

        print()
        top10 /= n
        top100 /= n
        mean_rank /= n
        ftop10 /= n
        ftop100 /= n
        fmean_rank /= n

        auc_x, auc_y, rank_auc = compute_rank_roc(ranks, len(filtered))
        auc_x, auc_y, frank_auc = compute_rank_roc(franks, len(filtered))
        df = pd.DataFrame({'auc_x': auc_x, 'auc_y': auc_y})
        df.to_pickle(gene_annots_file + '_auc.pkl')
        print(f'{top10:.2f} {top100:.2f} {mean_rank:.2f} {rank_auc:.2f}')
        print(f'{ftop10:.2f} {ftop100:.2f} {fmean_rank:.2f} {frank_auc:.2f}')


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(labels, preds):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def compute_rank_roc(ranks, n_prots):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_prots)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_prots
    return auc_x, auc_y, auc



if __name__ == '__main__':
    main()
