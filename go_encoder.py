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

from utils import Ontology, FUNC_DICT

logging.basicConfig(filename='logs.log', level=logging.DEBUG)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


@ck.command()
@ck.option(
    '--data-file', '-trdf', default='data/swissprot_exp.pkl',
    help='Data file with training features')
@ck.option(
    '--gos-file', '-gf', default='data/gos.pkl',
    help='GO class files')
@ck.option(
    '--model-file', '-mf', default='data/go_encoder.h5',
    help='DeepGOPlus model')
@ck.option(
    '--batch-size', '-bs', default=32,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=1024,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--logger-file', '-lf', default='data/encoder-training.csv',
    help='Batch size')
def main(data_file, gos_file, model_file, batch_size, epochs, load, logger_file):
    gos_df = pd.read_pickle(gos_file)
    gos = gos_df['gos'].values.flatten()
    gos_dict = {v: i for i, v in enumerate(gos)}

    params = {
        'input_length': len(gos),
        'loss': 'binary_crossentropy',
        'optimizer': 'adam'
    }
    train_df, valid_df, test_df = load_data(data_file)
    
    test_steps = int(math.ceil(len(test_df) / batch_size))
    test_generator = DFGenerator(test_df, gos_dict, batch_size)

    if load:
        print('Loading pretrained model')
        model = load_model(model_file)
    else:
        print('Creating a new model')
        model = create_model(params)

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
        train_generator = DFGenerator(train_df, gos_dict,
                                    batch_size)
        valid_generator = DFGenerator(valid_df, gos_dict,
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


def create_model(params):
    inp = Input(shape=(params['input_length'],), dtype=np.float32)
    net = Dense(2000, activation='relu')(inp)
    output = Dense(params['input_length'], activation='sigmoid')(net)

    model = Model(inputs=inp, outputs=output)
    model.summary()
    model.compile(
        optimizer=params['optimizer'],
        loss=params['loss'])
    logging.info('Compilation finished')
    return model


def load_data(data_file, split=0.9):
    df = pd.read_pickle(data_file)
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

class DFGenerator(Sequence):                                                                                                               
                                                                                                                                         
    def __init__(self, df, gos_dict, batch_size):
        self.start = 0
        self.size = len(df)
        self.df = df
        self.batch_size = batch_size
        self.gos_dict = gos_dict
                                                                                                                                         
    def __len__(self):                                                                                                                   
        return np.ceil(len(self.df) / float(self.batch_size)).astype(np.int32)                                                           
                                                                                                                                         
    def __getitem__(self, idx):                                                                                                          
        batch_index = np.arange(                                                                                                         
            idx * self.batch_size, min(self.size, (idx + 1) * self.batch_size))                                                          
        df = self.df.iloc[batch_index]                                                                                                   
        data_gos = np.zeros((len(df), len(self.gos_dict)), dtype=np.float32)
        # labels = np.zeros((len(df), len(self.terms_dict)), dtype=np.int32)
        for i, row in enumerate(df.itertuples()):
            for item in row.annotations:
                # t_id, score = item.split('|')
                if item in self.gos_dict:
                    data_gos[i, self.gos_dict[item]] = 1
        return (data_gos, data_gos)


if __name__ == '__main__':
    main()
