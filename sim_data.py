#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--hp-file', '-hf', default='data/hp.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--train-data-file', '-trdf', default='data/human.pkl',
    help='Data file with training features')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--dis-phenotypes', '-dp', default='data/phenotype_annotation.tab',
    help='Data file')
@ck.option(
    '--omim-file', '-dp', default='data/morbidmap.txt',
    help='Data file')
@ck.option(
    '--predictions-file', '-pf', default='data/predictions_max.pkl',
    help='Data file')
@ck.option(
    '--gene-annots-file', '-gaf', default='data/gene_annotations.tab',
    help='Gene Phenotype annotations')
@ck.option(
    '--dis-annots-file', '-daf', default='data/dis_annotations.tab',
    help='Disease Phenotype annotations')
@ck.option(
    '--fold', '-f', default=1,
    help='Fold index')
def main(hp_file, train_data_file, terms_file, dis_phenotypes, omim_file, predictions_file,
         gene_annots_file, dis_annots_file, fold):
    
    hp = Ontology(hp_file, with_rels=True)
    print('HP loaded')
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    predictions_file = f'fold{fold}_' + predictions_file
    gene_annots_file = f'fold{fold}_' + gene_annots_file
    dis_annots_file = f'fold{fold}_' + dis_annots_file
    real_annots_file = f'fold{fold}_data/gene_annotations_real.tab'
    
    diseases = set()
    genes = set()
    with open(omim_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            it = line.strip().split('\t')
            omim_id = it[0].split(', ')[-1].split()[0]
            gene_symbols = it[1].split(', ')
            genes |= set(gene_symbols)
            diseases.add('OMIM:' + omim_id)
    print(len(diseases), len(genes))
    dis_annots = {}
    with open(dis_phenotypes) as f:
        for line in f:
            it = line.strip().split('\t')
            dis_id = it[0] + ':' + it[1]
            if dis_id not in diseases:
                continue
            hp_id = it[4]
            if not hp.has_term(hp_id):
                continue
            if dis_id not in dis_annots:
                dis_annots[dis_id] = set()
            dis_annots[dis_id].add(hp_id)

    with open(dis_annots_file, 'w') as w:
        for dis_id, annots in dis_annots.items():
            w.write(dis_id)
            for hp_id in annots:
                w.write('\t' + hp_id)
            w.write('\n')
    
    df = pd.read_pickle(predictions_file)
    with open(gene_annots_file, 'w') as w:
        for i, row in df.iterrows():
            w.write(row['genes'])
            for hp_id in row['hp_preds']:
                w.write('\t' + hp_id)
            w.write('\n')

    with open(real_annots_file, 'w') as w:
        for i, row in df.iterrows():
            w.write(row['genes'])
            for hp_id in row['hp_annotations']:
                w.write('\t' + hp_id)
            w.write('\n')


    

if __name__ == '__main__':
    main()
