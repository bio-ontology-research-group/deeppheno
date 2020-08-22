#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import logging
import math
import time
import sys
import os
from collections import deque

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence

from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from deeppheno.aminoacids import MAXLEN, to_onehot
from deeppheno.utils import Ontology, FUNC_DICT, is_exp_code

from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel

logging.basicConfig(level=logging.DEBUG)


class HPOLayer(Layer):

    def __init__(self, nb_classes, **kwargs):
        self.nb_classes = nb_classes
        self.hpo_matrix = np.zeros((nb_classes, nb_classes), dtype=np.float32)
        super(HPOLayer, self).__init__(**kwargs)

    def set_hpo_matrix(self, hpo_matrix):
        self.hpo_matrix = hpo_matrix

    def get_config(self):
        config = super(HPOLayer, self).get_config()
        config['nb_classes'] = self.nb_classes
        return config
    
    def build(self, input_shape):
        self.kernel = K.variable(
            self.hpo_matrix, name='{}_kernel'.format(self.name))
        self.non_trainable_weights.append(self.kernel)
        super(HPOLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = tf.keras.backend.repeat(x, self.nb_classes)
        return tf.math.multiply(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.nb_classes, self.nb_classes] 


@ck.command()
@ck.option('--data-root', '-dr', default='data/', help='Data root folder', required=True)
@ck.option(
    '--in-file', '-if', required=True,
    help='Input file. TSV file with a list of genes with GO annotations (semicolon-space separated)')
@ck.option(
    '--hp-file', '-hf', default='hp.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--go-file', '-gof', default='go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--terms-file', '-tf', default='terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--gos-file', '-gf', default='gos.pkl',
    help='DataFrame with list of GO classes (as features)')
@ck.option(
    '--exp-file', '-ef', default='E-MTAB-5214-query-results.tpms.tsv',
    help='DataFrame with list of GO classes (as features)')
@ck.option(
    '--model-file', '-mf', default='model.h5',
    help='DeepPheno model')
@ck.option(
    '--out-file', '-o', default='predictions.tsv',
    help='Result file with predictions')
@ck.option(
    '--batch-size', '-bs', default=32,
    help='Batch size')
@ck.option(
    '--threshold', '-th', default=0.5,
    help='Prediction threshold')
def main(data_root, in_file, hp_file, go_file, terms_file, gos_file, exp_file,
         model_file, out_file, batch_size, threshold):
    # Check data folder and required files
    try:
        if os.path.exists(data_root):
            hp_file = os.path.join(data_root, hp_file)
            go_file = os.path.join(data_root, go_file)
            model_file = os.path.join(data_root, model_file)
            terms_file = os.path.join(data_root, terms_file)
            gos_file = os.path.join(data_root, gos_file)
            exp_file = os.path.join(data_root, exp_file)
            if not os.path.exists(go_file):
                raise Exception(f'Gene Ontology file ({go_file}) is missing!')
            if not os.path.exists(hp_file):
                raise Exception(f'Human Phenotype Ontology file ({hp_file}) is missing!')
            if not os.path.exists(model_file):
                raise Exception(f'Model file ({model_file}) is missing!')
            if not os.path.exists(terms_file):
                raise Exception(f'Terms file ({terms_file}) is missing!')
            if not os.path.exists(gos_file):
                raise Exception(f'GOs file ({gos_file}) is missing!')
            if not os.path.exists(exp_file):
                raise Exception(f'Expressions file ({exp_file}) is missing!')
        else:
            raise Exception(f'Data folder {data_root} does not exist!')
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    gos_df = pd.read_pickle(gos_file)
    gos = gos_df['gos'].values.flatten()
    gos_dict = {v: i for i, v in enumerate(gos)}

    global hpo
    hpo = Ontology(hp_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    global terms
    terms = terms_df['terms'].values.flatten()
    global term_set
    term_set = set(terms)
    df = load_data(in_file, exp_file)
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = len(terms)
    params = {}
    params['nb_classes'] = nb_classes
    print(len(terms_dict))
    steps = int(math.ceil(len(df) / batch_size))
    generator = DFGenerator(df, gos_dict, terms_dict,
                                 len(df))
    x, y = generator[0]

    print('Loading pretrained model')
    model = load_model(model_file, custom_objects={'HPOLayer': HPOLayer})
    model.summary()
    preds = model.predict(x)
    with open(out_file, 'w') as f:
        for i, row in enumerate(df.itertuples()):
            f.write(row.genes)
            for j in range(len(terms)):
                if preds[i, j] < 0.1:
                    continue
                f.write(f'\t{terms[j]}|{preds[i, j]:.3f}')
            f.write('\n')

def load_data(in_file, exp_file):
    gene_exp = {}
    with open(exp_file) as f:
        for line in f:
            if line.startswith('#') or line.startswith('Gene'):
                continue
            it = line.strip().split('\t')
            gene_name = it[1].split()[0].upper()
            exp = np.zeros((53,), dtype=np.float32)
            for i in range(len(it[2:])):
                exp[i] = float(it[2 + i]) if it[2 + i] != '' else 0.0
            gene_exp[gene_name] = exp / np.max(exp)
    annotations = []
    expressions = []
    genes = []
    with open(in_file) as f:
        for line in f:
            it = line.strip().split('\t')
            print(it)
            gene_name = it[0].upper()
            annots = it[1].split('; ')
            exp = np.zeros((53,), dtype=np.float32)
            if gene_name in gene_exp:
                exp = gene_exp[gene_name]
            genes.append(it[0])
            annotations.append(annots)
            expressions.append(exp)
    df = pd.DataFrame(
        {'genes': genes, 'annotations': annotations,
         'expressions': expressions})
            
    return df
    

class DFGenerator(Sequence):                                                                                                               
                                                                                                                                         
    def __init__(self, df, gos_dict, terms_dict, batch_size):
        self.start = 0
        self.size = len(df)
        self.df = df
        self.batch_size = batch_size
        self.terms_dict = terms_dict
        self.gos_dict = gos_dict
                                                                                                                                         
    def __len__(self):                                                                                                                   
        return np.ceil(len(self.df) / float(self.batch_size)).astype(np.int32)                                                           
                                                                                                                                         
    def __getitem__(self, idx):                                                                                                          
        batch_index = np.arange(                                                                                                         
            idx * self.batch_size, min(self.size, (idx + 1) * self.batch_size))                                                          
        df = self.df.iloc[batch_index]                                                                                                   
        data_gos = np.zeros((len(df), len(self.gos_dict)), dtype=np.float32)
        data_exp = np.zeros((len(df), 53), dtype=np.float32)
        labels = np.zeros((len(df), len(self.terms_dict)), dtype=np.int32)
        for i, row in enumerate(df.itertuples()):
            data_exp[i, :] = row.expressions
            for t_id in row.annotations:
                if t_id in self.gos_dict:
                    data_gos[i, self.gos_dict[t_id]] = 1

        data = [data_gos, data_exp]
        return (data, labels)
    
    
if __name__ == '__main__':
    main()
