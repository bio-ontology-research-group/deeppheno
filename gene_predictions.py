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
    '--gene', '-p', default='4200',
    help='Gene Entrez id')
def main(go_file, hp_file, terms_file, preds_file, gene):
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
    row = df.loc[df['genes'] == gene]
    
    with open(f'data/{gene}.deepgo_annotations.txt', 'w') as f:
        dg = [x.split('|') for x in row['deepgo_annotations'].values[0]]
        dg = sorted(dg, key=lambda x: float(x[1]), reverse=True)
        for go_id, score in dg:
            name = go.get_term(go_id)['name']
            f.write(f'{go_id}\t{name}\t{score}\n')

    with open(f'data/{gene}.go_annotations.txt', 'w') as f:
        dg = [x for x in row['go_annotations'].values[0]]
        for go_id in dg:
            name = go.get_term(go_id)['name']
            f.write(f'{go_id}\t{name}\n')

    with open(f'data/{gene}.deeppheno_annotations.txt', 'w') as f:
        dp = [(terms[i], score) for i, score in enumerate(row['preds'].values[0])]
        dp = sorted(dp, key=lambda x: x[1], reverse=True)
        for hp_id, score in dp:
            name = hp.get_term(hp_id)['name']
            f.write(f'{hp_id}\t{name}\t{score}\n')
            if score < 0.01:
                break




if __name__ == '__main__':
    main()
