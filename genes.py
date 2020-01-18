#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT
import logging
import json
import gzip

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--hp-file', '-hf', default='data/hp.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Terms for prediction')
@ck.option(
    '--preds-file', '-pf', default='data/all_predictions.pkl',
    help='Data file')
@ck.option(
    '--pheno', '-p', default='HP:0005978',
    help='Data file')
@ck.option(
    '--ukb-file', '-uf', default='data/E11_gwas_genes.pkl',
    help='Data file')
@ck.option(
    '--gwas-file', '-gf', default='data/gwas-association-downloaded_2019-10-09-EFO_0001360-withChildTraits.tsv',
    help='Data file')
def main(go_file, hp_file, terms_file, preds_file, pheno, ukb_file, gwas_file):
    go = Ontology(go_file, with_rels=True)
    print('GO loaded')
    hp = Ontology(hp_file, with_rels=True)
    print('HP loaded')

    terms_df = pd.read_pickle(terms_file)
    global terms
    terms = terms_df['terms'].values.flatten()
    labels = terms_df['labels'].values.flatten()
    print('Phenotypes', len(terms))
    global term_set
    term_set = set(terms)
    terms_dict = {v: i for i, v in enumerate(terms)}
    
    df = pd.read_pickle(preds_file)
    # res = []
    # for i, pheno in enumerate(terms):
    #     exp_genes = set()
    #     pred_genes = set()
    #     i = terms_dict[pheno]
    #     for row in df.itertuples():
    #         if pheno in row.hp_annotations:
    #             exp_genes.add(row.genes)
    #         if row.preds[i] >= 0.28:
    #             pred_genes.add(row.genes)

    #     inter = exp_genes.intersection(pred_genes)
    #     tp = len(inter)
    #     fp = len(pred_genes) - tp
    #     if (fp + tp > 0):
    #         p = tp / (fp + tp)
    #     res.append((p, pheno, len(exp_genes), len(inter), len(pred_genes), labels[i]))
    # res = sorted(res, key=lambda x: x[0], reverse=True)
    # for it in res:
    #     print('{0:.3f}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(*it))
    # return
    
    exp_genes = set()
    pred_genes = set()
    i = terms_dict[pheno]
    for row in df.itertuples():
        if pheno in row.hp_annotations:
            exp_genes.add(row.genes)
        if row.preds[i] >= 0.28:
            pred_genes.add(row.genes)

    fgenes = pred_genes - exp_genes
    both = exp_genes.intersection(pred_genes)
    print(len(exp_genes), len(fgenes), len(pred_genes), len(both))
    igenes = {}
    inters = load_interactions()
    for g in exp_genes:
        if g in inters:
            for gn in inters[g]:
                if gn not in igenes:
                    igenes[gn] = set()
                igenes[gn].add(g)
    gene_names = {}
    df = pd.read_pickle('data/swissprot.pkl')
    for row in df.itertuples():
        gene_names[row.genes] = row.gene_names
    genes = set()
    inter_genes = set()
    for g in fgenes:
        #genes |= set(gene_names[g])
        genes.add(gene_names[g][0])
        if g in igenes:
            name = gene_names[g][0]
            # for gn in igenes[g]:
            #     if gn in gene_names:
            #         name += '_' + gene_names[gn][0]
            #     else:
            #         name += '_' + gn
            inter_genes.add(name)
    print(json.dumps(list(genes)))
    print(json.dumps(list(inter_genes)))
    print(len(fgenes), len(fgenes.intersection(igenes)))
    # res = sorted(res, key=lambda x: x[0], reverse=True)
    # for item in res:
    #     print(item)

    # UKBIOBank GWAS file
    df = pd.read_pickle(ukb_file)
    ukb_genes = set(df['genes'].values)

    gwas_genes = set()
    df = pd.read_csv(gwas_file, sep='\t', encoding='utf-8')
    for it in df['MAPPED_GENE']:
        mapped_genes = str(it).replace(' - ', ', ')
        gwas_genes |= set(mapped_genes.split(', '))

    print('UKBGenes', genes.intersection(ukb_genes))
    print('GWASGenes', genes.intersection(gwas_genes))

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

if __name__ == '__main__':
    main()
