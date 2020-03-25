#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from subprocess import Popen, PIPE
import time
from utils import Ontology, NAMESPACES
from aminoacids import to_onehot
import gzip

MAXLEN = 2000

@ck.command()
@ck.option('--in-file', '-if', default='data/all_predictions.pkl', help='file', required=True)
@ck.option('--hp-file', '-hf', default='data/hp.obo', help='HP Ontology')
@ck.option('--terms-file', '-tf', default='data/terms.pkl', help='List of predicted terms')
@ck.option('--out-file', '-of', default='data/out.txt', help='result file')
@ck.option('--map-file', '-mf', default='data/tar2prot.txt', help='map file')
def main(in_file, hp_file, terms_file, out_file, map_file):
    # Load GO and read list of all terms
    hp = Ontology(hp_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    df = pd.read_pickle(in_file)
    mapping = {}
    with open(map_file) as f:
        for line in f:
            it = line.strip().split()
            mapping[it[1]] = it[0]

    w = open(out_file, 'w')
    # w.write('AUTHOR Hoehndorf Lab - DeepGO team\n')
    # w.write('MODEL 1\n')
    # w.write('KEYWORDS machine learning, sequence alignment.\n')
    for row in df.itertuples():
        prot_id = row.genes
        # if prot_id not in mapping:
        #     continue
        # prot_id = mapping[prot_id]
        for i, score in enumerate(row.preds):
            if score >= 0.1:
                w.write(prot_id + '\t' + terms[i] + '\t%.2f\n' % score)
    # w.write('END\n')
    w.close()

if __name__ == '__main__':
    main()
