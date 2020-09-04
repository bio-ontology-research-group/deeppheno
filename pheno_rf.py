#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging
import math
import time
from collections import deque

from tensorflow.keras.utils import Sequence
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, accuracy_score
from aminoacids import MAXLEN, to_onehot
from utils import Ontology, FUNC_DICT, is_exp_code
from joblib import dump, load

logging.basicConfig(level=logging.DEBUG)


@ck.command()
@ck.option(
    '--hp-file', '-hf', default='data/hp.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/human.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--gos-file', '-gf', default='data/gos.pkl',
    help='DataFrame with list of GO classes (as features)')
@ck.option(
    '--out-file', '-o', default='data/predictions.pkl',
    help='Result file with predictions for test set')
@ck.option(
    '--fold', '-f', default=1,
    help='Fold index')
@ck.option(
    '--batch-size', '-bs', default=32,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=1024,
    help='Training epochs')
@ck.option(
    '--load_model', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--logger-file', '-lf', default='data/training.csv',
    help='Batch size')
@ck.option(
    '--threshold', '-th', default=0.5,
    help='Prediction threshold')
@ck.option(
    '--device', '-d', default='gpu:1',
    help='Prediction threshold')
@ck.option(
    '--estimators', '-es', default=10,
    help='Random forest n_estimators')
def main(hp_file, data_file, terms_file, gos_file,
         out_file, fold, batch_size, epochs, load_model, logger_file, threshold,
         device, estimators):
    gos_df = pd.read_pickle(gos_file)
    gos = gos_df['gos'].values.flatten()
    gos_dict = {v: i for i, v in enumerate(gos)}

    # cross validation settings
    out_file = f'fold{fold}_exp-' + out_file
    params = {'n_estimators': estimators}
    print('Params:', params)
    global hpo
    hpo = Ontology(hp_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    global terms
    terms = terms_df['terms'].values.flatten()
    print('Phenotypes', len(terms))
    global term_set
    term_set = set(terms)
    train_df, valid_df, test_df = load_data(data_file, terms, fold)
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = len(terms)
    params['nb_classes'] = nb_classes
    print(len(terms_dict))
    test_steps = int(math.ceil(len(test_df) / batch_size))
    test_generator = DFGenerator(test_df, gos_dict, terms_dict,
                                 len(test_df))
    valid_steps = int(math.ceil(len(valid_df) / batch_size))
    train_steps = int(math.ceil(len(train_df) / batch_size))

    xy_generator = DFGenerator(train_df, gos_dict, terms_dict,
                                  len(train_df))
    x, y = xy_generator[0]
    val_generator = DFGenerator(valid_df, gos_dict, terms_dict,
                                  len(valid_df))
    val_x, val_y = val_generator[0]
    test_x, test_y = test_generator[0]
    if load_model:
        logging.info(f'Loading RandomForest_{estimators} classifier')
        clf = load(f'data/rf_{estimators}.joblib')
    else:
        logging.info('Training RandomForest classifier')
        clf = RandomForestRegressor(n_estimators=params['n_estimators'])
        clf.fit(x, y)
        dump(clf, f'data/rf_{estimators}.joblib')
    
    logging.info('Evaluating model')
    val_preds = clf.predict(val_x)
    # val_accuracy = accuracy_score(val_preds, val_y)
    # print('Val accuracy', val_accuracy)

    preds = clf.predict(test_x)
    # test_accuracy = accuracy_score(preds, test_y)
    # print('Test accuracy', test_accuracy)

    all_terms_df = pd.read_pickle('data/all_terms.pkl')
    all_terms = all_terms_df['terms'].values
    all_terms_dict = {v:k for k,v in enumerate(all_terms)}
    all_labels = np.zeros((len(test_df), len(all_terms)), dtype=np.int32)
    for i, row in enumerate(test_df.itertuples()):
        for hp_id in row.hp_annotations:
            if hp_id in all_terms_dict:
                all_labels[i, all_terms_dict[hp_id]] = 1

    all_preds = np.zeros((len(test_df), len(all_terms)), dtype=np.float32)
    for i in range(len(test_df)):
        for j in range(nb_classes):
            all_preds[i, all_terms_dict[terms[j]]] = preds[i, j]
    logging.info('Computing performance:')
    roc_auc = compute_roc(all_labels, all_preds)
    print('ROC AUC: %.2f' % (roc_auc,))
    test_df['preds'] = list(preds)
    print(test_df)
    logging.info('Saving predictions')
    test_df.to_pickle(out_file)

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def load_data(data_file, terms, fold=1):
    df = pd.read_pickle(data_file)
    # Split train/valid
    n = len(df)
    index = np.arange(n)
    np.random.seed(seed=10)
    np.random.shuffle(index)
    index = list(index)
    train_index = []
    test_index = []
    fn = n / 5
    # 5 fold cross-validation
    for i in range(1, 6):
        start = int((i - 1) * fn)
        end = int(i * fn)
        if i == fold:
            test_index += index[start:end]
        else:
            train_index += index[start:end]
    assert n == len(test_index) + len(train_index)
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    valid_n = int(len(train_df) * 0.9)
    valid_df = train_df.iloc[valid_n:]
    train_df = train_df.iloc[:valid_n]
     
    # All Swissprot proteins
    # train_n = int(n * 0.9)
    # train_df = df.iloc[index[:train_n]]
    # valid_df = df.iloc[index[train_n:]]
    # test_df = pd.read_pickle('data/human_all.pkl')
    
    # CAFA2 Test data
    # train_n = int(n * 0.9)
    # train_df = df.iloc[index[:train_n]]
    # valid_df = df.iloc[index[train_n:]]
    # test_df = pd.read_pickle('data-cafa/human_test.pkl')
    print(len(df), len(train_df), len(valid_df), len(test_df))
    return train_df, valid_df, test_df
    

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
        data_seq = np.zeros((len(df), MAXLEN, 21), dtype=np.float32)
        data_gos = np.zeros((len(df), len(self.gos_dict)), dtype=np.float32)
        data_exp = np.zeros((len(df), 53), dtype=np.float32)
        labels = np.zeros((len(df), len(self.terms_dict)), dtype=np.int32)
        for i, row in enumerate(df.itertuples()):
            data_seq[i, :] = to_onehot(row.sequences)
            data_exp[i, :] = row.expressions
            for item in row.deepgo_annotations:
                t_id, score = item.split('|')
                if t_id in self.gos_dict:
                    data_gos[i, self.gos_dict[t_id]] = float(score)

            for t_id in row.iea_annotations:
                if t_id in self.gos_dict:
                    data_gos[i, self.gos_dict[t_id]] = 1

            for t_id in row.go_annotations:
                if t_id in self.gos_dict:
                    data_gos[i, self.gos_dict[t_id]] = 1
                
            for t_id in row.hp_annotations:
                if t_id in self.terms_dict:
                    labels[i, self.terms_dict[t_id]] = 1
        data = np.concatenate([data_gos, data_exp], axis=1)
        return (data, labels)
    
    
if __name__ == '__main__':
    main()
