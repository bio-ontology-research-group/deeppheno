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
from scipy.stats import rankdata, norm
import gzip


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def load_interactions():
    idmap = {}
    with open('data/string2gene.tsv') as f:
        for line in f:
            it = line.strip().split('\t')
            idmap[it[0]] = it[1]

    inter = {}
    with gzip.open('data/9606.protein.links.v11.0.txt.gz', 'rt') as f:
        next(f)
        for line in f:
            it = line.strip().split()
            score = int(it[2])
            if score < 700 or it[0] not in idmap or it[1] not in idmap:
                continue
            g1 = idmap[it[0]]
            g2 = idmap[it[1]]
            if g1 not in inter:
                inter[g1] = set()
            inter[g1].add(g2)
    return inter

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--hp-file', '-hf', default='data/hp.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--preds-file', '-pf', default='data/all_predictions.pkl',
    help='Data file with similarity values')
@ck.option(
    '--terms-file', '-tf', default='data/spec_terms.pkl',
    help='List of specific terms')
def main(go_file, hp_file, preds_file, terms_file):
    go = Ontology(go_file, with_rels=True)
    print('GO loaded')
    hp = Ontology(hp_file, with_rels=True)
    print('HP loaded')

    terms_df = pd.read_pickle(terms_file)
    global terms
    terms = terms_df['terms'].values.flatten()
    labels = terms_df['labels'].values.flatten()
    index = terms_df['index'].values.flatten()
    print('Phenotypes', len(terms))
    global term_set
    term_set = set(terms)
    terms_dict = {v: i for i, v in zip(index, terms)}
    
    df = pd.read_pickle(preds_file)
    inters = load_interactions()
    genes = df['genes'].values
    
    res = 0
    total = 0
    data = []
    with ck.progressbar(terms_dict.items()) as bar:
        for pheno, i in bar:
            exp_genes = set()
            pred_genes = set()
            for row in df.itertuples():
                if pheno in row.hp_annotations:
                    exp_genes.add(row.genes)
                if row.preds[i] >= 0.28:
                    pred_genes.add(row.genes)
            fgenes = pred_genes - exp_genes
            igenes = set()
            for g in exp_genes:
                if g in inters:
                    igenes |= inters[g]
            overlap = fgenes.intersection(igenes)
            tn = len(pred_genes)
            n = len(fgenes)
            m = len(igenes)
            x = len(overlap)
            if n > 0:
                o = x / n
                res += o
                # Random simulation
                a = np.zeros(1000, dtype=np.float32)
                for i in range(1000):
                    sim_genes = set(np.random.choice(genes, size=tn))
                    fgenes = sim_genes - exp_genes
                    overlap = fgenes.intersection(igenes)
                    if len(fgenes) > 0:
                        a[i] = len(overlap) / len(fgenes)
                    else:
                        a[i] = 0
                data.append(a)

    total = len(data)
    data = np.hstack(data).reshape(total, 1000)
    data = data.mean(axis=1)
    res /= total
    mean, std = data.mean(), data.std()
    print(res, mean, std)
    p = 1 - norm.cdf(res, mean, std)
    print('P-value', p)

if __name__ == '__main__':
    main()
