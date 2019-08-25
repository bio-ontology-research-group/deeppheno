#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import math
from collections import deque

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Conv1D, Flatten, Concatenate,
    MaxPooling1D, Dropout, Maximum
)
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.python.ops import math_ops
from tensorflow.keras import regularizers

from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from aminoacids import MAXLEN, to_onehot
from utils import Ontology, FUNC_DICT, is_exp_code

logging.basicConfig(filename='logs.log', level=logging.DEBUG)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


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
    '--model-file', '-mf', default='data/model.h5',
    help='DeepGOPlus model')
@ck.option(
    '--out-file', '-o', default='data/predictions.pkl',
    help='Result file with predictions for test set')
@ck.option(
    '--split', '-s', default=0.9,
    help='train/valid split')
@ck.option(
    '--batch-size', '-bs', default=32,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=1024,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--logger-file', '-lf', default='data/training.csv',
    help='Batch size')
@ck.option(
    '--threshold', '-th', default=0.5,
    help='Prediction threshold')
@ck.option(
    '--device', '-d', default='gpu:0',
    help='Prediction threshold')
@ck.option(
    '--params-index', '-pi', default=-1,
    help='Definition mapping file')
def main(hp_file, data_file, terms_file, gos_file, model_file,
         out_file, split, batch_size, epochs, load, logger_file, threshold,
         device, params_index):
    gos_df = pd.read_pickle(gos_file)
    gos = gos_df['gos'].values.flatten()
    gos_dict = {v: i for i, v in enumerate(gos)}
    
    params = {
        'input_shape': (len(gos),),
        'optimizer': Adam(0.0001),
        'loss': 'binary_crossentropy'
    }
    
    print('Params:', params)
    global hp
    hp = Ontology(hp_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    global terms
    terms = terms_df['terms'].values.flatten()
    print('Phenotypes', len(terms))
    global term_set
    term_set = set(terms)
    train_df, valid_df, test_df = load_data(data_file, terms, split)
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = len(terms)
    print(len(terms_dict))
    test_steps = int(math.ceil(len(test_df) / batch_size))
    test_generator = DFGenerator(test_df, gos_dict, terms_dict,
                                 batch_size)
    if load:
        print('Loading pretrained model')
        model = load_model(model_file)
    else:
        print('Creating a new model')
        model = create_model(nb_classes, params)

        print("Training data size: %d" % len(train_df))
        print("Validation data size: %d" % len(valid_df))
        checkpointer = ModelCheckpoint(
            filepath=model_file,
            verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
        logger = CSVLogger(logger_file)

        print('Starting training the model')

        valid_steps = int(math.ceil(len(valid_df) / batch_size))
        train_steps = int(math.ceil(len(train_df) / batch_size))
        train_generator = DFGenerator(train_df, gos_dict, terms_dict,
                                    batch_size)
        valid_generator = DFGenerator(valid_df, gos_dict, terms_dict,
                                      batch_size)

        model.summary()
        # print(model.layers[1].get_weights()[0].shape)
        # return
        model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            max_queue_size=batch_size,
            workers=12,
            callbacks=[logger, checkpointer, earlystopper])
        logging.info('Loading best model')
        model = load_model(model_file)


    logging.info('Evaluating model')
    loss = model.evaluate_generator(test_generator, steps=test_steps)
    print('Test loss %f' % loss)

    logging.info('Predicting')
    preds = model.predict_generator(test_generator, steps=test_steps, verbose=1)

    test_labels = np.zeros((len(test_df), nb_classes), dtype=np.int32)
    for i, row in enumerate(test_df.itertuples()):
        for hp_id in row.hp_annotations:
            if hp_id in terms_dict:
                test_labels[i, terms_dict[hp_id]] = 1
    logging.info('Computing performance:')
    roc_auc = compute_roc(test_labels, preds)
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


def get_node(node_id, net):
    name = node_id.split(':')[1]
    net = Dense(1, activation='sigmoid')(net)
    return net

def get_layers(inputs):
    q = deque()
    layers = {}
    ROOT = 'HP:0000001'
    layers[ROOT] = {'net': inputs}
    for node_id in hp.get_children(ROOT):
        if node_id in term_set:
            q.append((node_id, inputs))
    while len(q) > 0:
        node_id, net = q.popleft()
        output = get_node(node_id, net)
        if node_id not in layers:
            layers[node_id] = {'net': net, 'output': output}
            for n_id in hp.get_children(node_id):
                if n_id in term_set and n_id not in layers:
                    q.append((n_id, output))

    for node_id in terms:
        childs = set(hp.get_children(node_id)).intersection(term_set)
        if len(childs) > 0:
            outputs = [layers[node_id]['output']]
            for ch_id in childs:
                outputs.append(layers[ch_id]['output'])
            name = node_id.split(':')[1] + '_max'
            layers[node_id]['output'] = Maximum(name=name)(outputs)

    return layers


def create_model(nb_classes, params):
    inp = Input(shape=params['input_shape'], dtype=np.float32)
    net = Dense(1000, name='dense', activation='relu')(inp)
    output = Dense(
        nb_classes, activation='sigmoid',
        name='dense_out')(net)

    model = Model(inputs=inp, outputs=output)
    model.summary()
    model.compile(
        optimizer=params['optimizer'],
        loss=params['loss'])
    logging.info('Compilation finished')

    return model



def load_data(data_file, terms, split):
    df = pd.read_pickle(data_file)
    n = len(df)
    # Split train/valid
    n = len(df)
    index = np.arange(n)
    train_n = int(n * split)
    valid_n = int(train_n * split)
    np.random.seed(seed=10)
    np.random.shuffle(index)
    train_df = df.iloc[index[:valid_n]]
    valid_df = df.iloc[index[valid_n:train_n]]
    test_df = df.iloc[index[train_n:]]
    # CAFA2 Test data
    # train_df = df.iloc[index[:train_n]]
    # valid_df = df.iloc[index[train_n:]]
    # test_df = pd.read_pickle('data/human_test.pkl')
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
        labels = np.zeros((len(df), len(self.terms_dict)), dtype=np.int32)
        for i, row in enumerate(df.itertuples()):
            data_seq[i, :] = to_onehot(row.sequences)
            
            for item in row.deepgo_annotations:
                t_id, score = item.split('|')
                if t_id in self.gos_dict:
                    data_gos[i, self.gos_dict[t_id]] = float(score)

            # for t_id in row.iea_annotations:
            #     if t_id in self.gos_dict:
            #         data_gos[i, self.gos_dict[t_id]] = 1

            # for t_id in row.go_annotations:
            #     if t_id in self.gos_dict:
            #         data_gos[i, self.gos_dict[t_id]] = 1
                
            for t_id in row.hp_annotations:
                if t_id in self.terms_dict:
                    labels[i, self.terms_dict[t_id]] = 1
        return (data_gos, labels)
    
    
if __name__ == '__main__':
    main()
