#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import math

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Conv1D, Flatten, Concatenate,
    MaxPooling1D, Dropout,
)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

from utils import Ontology, FUNC_DICT

logging.basicConfig(level=logging.INFO)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

@ck.command()
@ck.option(
    '--mp-file', '-mpf', default='data/mp.obo',
    help='Mouse Phenotype Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/mouse.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--gos-file', '-gf', default='data/gos.pkl',
    help='DataFrame with list of GO classes (as features)')
@ck.option(
    '--model-file', '-mf', default='data/mp_model.h5',
    help='DeepGOPlus model')
@ck.option(
    '--out-file', '-o', default='data/mp_predictions.pkl',
    help='Result file with predictions for test set')
@ck.option(
    '--split', '-s', default=0.9,
    help='train/test split')
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
def main(mp_file, data_file, terms_file, gos_file, model_file,
         out_file, split, batch_size, epochs, load, logger_file, threshold,
         device, params_index):
    gos_df = pd.read_pickle(gos_file)
    gos = gos_df['gos'].values.flatten()
    gos_dict = {v: i for i, v in enumerate(gos)}
    
    params = {
        'input_length': len(gos),
        'optimizer': Adam(lr=1e-4),
        'loss': 'binary_crossentropy'
    }
    
    print('Params:', params)
    
    mp = Ontology(mp_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    
    train_df, valid_df, test_df = load_data(data_file, terms, split)
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = len(terms)

    with tf.device('/' + device):
        test_steps = int(math.ceil(len(test_df) / batch_size))
        test_generator = DFGenerator(test_df, gos_dict, terms_dict,
                                     batch_size)
        if load:
            logging.info('Loading pretrained model')
            model = load_model(model_file)
        else:
            logging.info('Creating a new model')
            model = create_model(nb_classes, params)
            
            logging.info("Training data size: %d" % len(train_df))
            logging.info("Validation data size: %d" % len(valid_df))
            checkpointer = ModelCheckpoint(
                filepath=model_file,
                verbose=1, save_best_only=True)
            earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
            logger = CSVLogger(logger_file)

            logging.info('Starting training the model')

            valid_steps = int(math.ceil(len(valid_df) / batch_size))
            train_steps = int(math.ceil(len(train_df) / batch_size))
            train_generator = DFGenerator(train_df, gos_dict, terms_dict,
                                        batch_size)
            valid_generator = DFGenerator(valid_df, gos_dict, terms_dict,
                                          batch_size)
    
            model.summary()
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
        logging.info('Test loss %f' % loss)
        
        logging.info('Predicting')
        test_generator.reset()
        preds = model.predict_generator(test_generator, steps=test_steps)
    
    test_labels = np.zeros((len(test_df), nb_classes), dtype=np.int32)
    for i, row in enumerate(test_df.itertuples()):
        for hp_id in row.mp_annotations:
            if hp_id in terms_dict:
                test_labels[i, terms_dict[hp_id]] = 1
    logging.info('Computing performance:')
    roc_auc = compute_roc(test_labels, preds)
    logging.info('ROC AUC: %.2f' % (roc_auc,))
    test_df['preds'] = list(preds)
    
    logging.info('Saving predictions')
    test_df.to_pickle(out_file)


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def create_model(nb_classes, params):
    inp = Input(shape=(params['input_length'],), dtype=np.float32)
    net = Dense(nb_classes, activation='relu', name='net1')(inp)
    output = Dense(nb_classes, activation='sigmoid', name='dense_out')(net)

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
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = df.iloc[index[:valid_n]]
    valid_df = df.iloc[index[valid_n:train_n]]
    test_df = df.iloc[index[train_n:]]
    return train_df, valid_df, test_df
    

class DFGenerator(object):

    def __init__(self, df, gos_dict, terms_dict, batch_size):
        self.start = 0
        self.size = len(df)
        self.df = df
        self.batch_size = batch_size
        self.terms_dict = terms_dict
        self.gos_dict = gos_dict
        
    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            df = self.df.iloc[batch_index]
            data = np.zeros((len(df), len(self.gos_dict)), dtype=np.float32)
            labels = np.zeros((len(df), len(self.terms_dict)), dtype=np.int32)
            for i, row in enumerate(df.itertuples()):
                # for item in row.deepgo_annotations:
                #     t_id, score = item.split('|')
                #     if t_id in self.gos_dict:
                #         data[i, self.gos_dict[t_id]] = float(score)
                for t_id in row.go_annotations:
                    if t_id in self.gos_dict:
                        data[i, self.gos_dict[t_id]] = 1
                for t_id in row.mp_annotations:
                    if t_id in self.terms_dict:
                        labels[i, self.terms_dict[t_id]] = 1
            self.start += self.batch_size
            return (data, labels)
        else:
            self.reset()
            return self.next()

    
if __name__ == '__main__':
    main()
