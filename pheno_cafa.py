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
    MaxPooling1D, Dropout, Maximum, Layer
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

from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel

logging.basicConfig(level=logging.DEBUG)

print("GPU Available: ", tf.test.is_gpu_available())


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
        assert input_shape[1] == self.nb_classes
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
@ck.option(
    '--hp-file', '-hf', default='data-cafa/hp.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data-cafa/human.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--terms-file', '-tf', default='data-cafa/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--gos-file', '-gf', default='data-cafa/gos.pkl',
    help='DataFrame with list of GO classes (as features)')
@ck.option(
    '--model-file', '-mf', default='data-cafa/model.h5',
    help='DeepGOPlus model')
@ck.option(
    '--out-file', '-o', default='data-cafa/predictions.pkl',
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
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--logger-file', '-lf', default='data-cafa/training.csv',
    help='Batch size')
@ck.option(
    '--threshold', '-th', default=0.5,
    help='Prediction threshold')
@ck.option(
    '--device', '-d', default='gpu:1',
    help='Prediction threshold')
def main(hp_file, data_file, terms_file, gos_file, model_file,
         out_file, fold, batch_size, epochs, load, logger_file, threshold,
         device):
    gos_df = pd.read_pickle(gos_file)
    gos = gos_df['gos'].values.flatten()
    gos_dict = {v: i for i, v in enumerate(gos)}

    # cross validation settings
    # model_file = f'fold{fold}_' + model_file
    # out_file = f'fold{fold}_' + out_file
    params = {
        'input_shape': (len(gos),),
        'exp_shape': 53,
        'nb_layers': 1,
        'loss': 'binary_crossentropy',
        'rate': 0.3,
        'learning_rate': 0.001,
        'units': 1500, # 750
        'model_file': model_file
    }
    
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
    hpo_matrix = get_hpo_matrix(hpo, terms_dict)
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

    # train_generator = DFGenerator(train_df, gos_dict, terms_dict,
    #                               batch_size)
    # valid_generator = DFGenerator(valid_df, gos_dict, terms_dict,
    #                               batch_size)
    
    with tf.device(device):
        if load:
            print('Loading pretrained model')
            model = load_model(model_file, custom_objects={'HPOLayer': HPOLayer})
            flat_model = load_model(model_file + '_flat.h5')
        else:
            print('Creating a new model')
            flat_model = MyHyperModel(params)
            # flat_model = create_flat_model(params)

            print("Training data size: %d" % len(train_df))
            print("Validation data size: %d" % len(valid_df))
            checkpointer = ModelCheckpoint(
                filepath=model_file + '_flat.h5',
                verbose=1, save_best_only=True)
            earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
            logger = CSVLogger(logger_file)

            # print('Starting training the flat model')
            # flat_model.summary()
            # flat_model.fit(
            #     train_generator,
            #     steps_per_epoch=train_steps,
            #     epochs=epochs,
            #     validation_data=valid_generator,
            #     validation_steps=valid_steps,
            #     max_queue_size=batch_size,
            #     workers=12,
            #     callbacks=[checkpointer, earlystopper])

            tuner = RandomSearch(
                flat_model,
                objective='val_loss',
                max_trials=50,
                directory='data-cafa',
                project_name='pheno')
            tuner.search(
                x, y, epochs=100, validation_data=(val_x, val_y),
                callbacks=[earlystopper])
            tuner.results_summary()
            logging.info('Loading best model')
            flat_model = tuner.get_best_models(num_models=1)[0]
            flat_model.summary()
            loss = flat_model.evaluate(val_x, val_y)
            print('Valid loss %f' % loss)
            flat_model.save(model_file + '_flat.h5')

            model = create_model(params, hpo_matrix)

            checkpointer = ModelCheckpoint(
                filepath=model_file,
                verbose=1, save_best_only=True)
            model.summary()
            print('Starting training the model')
            model.fit(
                x, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(val_x, val_y),
                callbacks=[logger, checkpointer, earlystopper])

            logging.info('Loading best model')
            model = load_model(model_file, custom_objects={'HPOLayer': HPOLayer})
            flat_model = load_model(model_file + '_flat.h5')
            
        logging.info('Evaluating model')
        loss = flat_model.evaluate(test_x, test_y, batch_size=batch_size)
        print('Flat Test loss %f' % loss)
        loss = model.evaluate(test_x, test_y, batch_size=batch_size)
        print('Test loss %f' % loss)

        logging.info('Predicting')
        preds = model.predict(test_x, batch_size=batch_size, verbose=1)
        flat_preds = flat_model.predict(test_x, batch_size=batch_size, verbose=1)

        all_terms_df = pd.read_pickle('data-cafa/all_terms.pkl')
        all_terms = all_terms_df['terms'].values
        all_terms_dict = {v:k for k,v in enumerate(all_terms)}
        all_labels = np.zeros((len(test_df), len(all_terms)), dtype=np.int32)
        for i, row in enumerate(test_df.itertuples()):
            for hp_id in row.hp_annotations:
                if hp_id in all_terms_dict:
                    all_labels[i, all_terms_dict[hp_id]] = 1
        
        all_preds = np.zeros((len(test_df), len(all_terms)), dtype=np.float32)
        all_flat_preds = np.zeros((len(test_df), len(all_terms)), dtype=np.float32)
        for i in range(len(test_df)):
            for j in range(nb_classes):
                all_preds[i, all_terms_dict[terms[j]]] = preds[i, j]
                all_flat_preds[i, all_terms_dict[terms[j]]] = flat_preds[i, j]
        logging.info('Computing performance:')
        roc_auc = compute_roc(all_labels, all_preds)
        print('ROC AUC: %.2f' % (roc_auc,))
        flat_roc_auc = compute_roc(all_labels, all_flat_preds)
        print('FLAT ROC AUC: %.2f' % (flat_roc_auc,))
        test_df['preds'] = list(preds)
        print(test_df)
        logging.info('Saving predictions')
        test_df.to_pickle(out_file)

        test_df['preds'] = list(flat_preds)
        test_df.to_pickle(out_file + '_flat.pkl')

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
    # train_index = []
    # test_index = []
    # fn = n / 5
    # 5 fold cross-validation
    # for i in range(1, 6):
    #     start = int((i - 1) * fn)
    #     end = int(i * fn)
    #     if i == fold:
    #         test_index += index[start:end]
    #     else:
    #         train_index += index[start:end]
    # assert n == len(test_index) + len(train_index)
    # train_df = df.iloc[train_index]
    # test_df = df.iloc[test_index]

    # valid_n = int(len(train_df) * 0.9)
    # valid_df = train_df.iloc[valid_n:]
    # train_df = train_df.iloc[:valid_n]
     
    # all Swissprot proteins
    # train_n = int(n * 0.9)
    # train_df = df.iloc[index[:train_n]]
    # valid_df = df.iloc[index[train_n:]]
    # test_df = pd.read_pickle('data-cafa/human_all.pkl')
    
    # CAFA2 Test data
    train_n = int(n * 0.9)
    train_df = df.iloc[index[:train_n]]
    valid_df = df.iloc[index[train_n:]]
    test_df = pd.read_pickle('data-cafa/human_test.pkl')
    print(len(df), len(train_df), len(valid_df), len(test_df))
    return train_df, valid_df, test_df
    

class MyHyperModel(HyperModel):

    def __init__(self, params):
        self.params = params

    def build(self, hp):
        inp = Input(shape=self.params['input_shape'], dtype=np.float32)
        exp_inp = Input(shape=self.params['exp_shape'], dtype=np.float32)
        net = inp
        for i in range(self.params['nb_layers']):
            net = Dense(
                units=hp.Int(
                    'units', min_value=250, max_value=2000, step=250),
                name=f'dense_{i}', activation='relu')(net)
            net = Dropout(hp.Choice('rate', values=[0.3, 0.5]))(net)
        net = Concatenate(axis=1)([net, exp_inp])
        output = Dense(
            self.params['nb_classes'], activation='sigmoid',
            name='dense_out')(net)

        model = Model(inputs=[inp, exp_inp], outputs=output)
        model.summary()
        model.compile(
            optimizer=Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            loss=self.params['loss'])
        return model


def get_hpo_matrix(hpo, terms_dict):
    nb_classes = len(terms_dict)
    res = np.zeros((nb_classes, nb_classes), dtype=np.float32)
    for hp_id, i in terms_dict.items():
        subs = hpo.get_term_set(hp_id)
        res[i, i] = 1
        for h_id in subs:
            if h_id in terms_dict:
                res[i, terms_dict[h_id]] = 1
    return res


def create_flat_model(params):
    inp = Input(shape=params['input_shape'], dtype=np.float32)
    exp_inp = Input(shape=self.params['exp_shape'], dtype=np.float32)
    net = inp
    for i in range(params['nb_layers']):
        net = Dense(
            units=params['units'], name=f'dense_{i}', activation='relu')(net)
        net = Dropout(rate=params['rate'])(net)
    net = Concatenate(axis=1)([net, exp_inp])
    net = Dense(
        params['nb_classes'], activation='sigmoid',
        name='dense_out')(net)
    output = Flatten()(net)
    model = Model(inputs=[inp, exp_inp], outputs=output)
    model.summary()
    model.compile(
        optimizer=Adam(lr=params['learning_rate']),
        loss=params['loss'])
    logging.info('Compilation finished')

    return model


def create_model(params, hpo_matrix):
    inp = Input(shape=params['input_shape'], dtype=np.float32)
    exp_inp = Input(shape=params['exp_shape'], dtype=np.float32)
    # Load flat model
    flat_model = load_model(params['model_file'] + '_flat.h5')
    net = flat_model([inp, exp_inp])
    hpo_layer = HPOLayer(params['nb_classes'])
    hpo_layer.trainable = False
    hpo_layer.set_hpo_matrix(hpo_matrix)
    net = hpo_layer(net)
    net = MaxPooling1D(pool_size=params['nb_classes'])(net)
    output = Flatten()(net)
    model = Model(inputs=[inp, exp_inp], outputs=output)
    model.summary()
    model.compile(
        optimizer=Adam(lr=params['learning_rate']),
        loss=params['loss'])
    logging.info('Compilation finished')

    return model



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

            # for t_id in row.iea_annotations:
            #     if t_id in self.gos_dict:
            #         data_gos[i, self.gos_dict[t_id]] = 1

            # for t_id in row.go_annotations:
            #     if t_id in self.gos_dict:
            #         data_gos[i, self.gos_dict[t_id]] = 1
                
            for t_id in row.hp_annotations:
                if t_id in self.terms_dict:
                    labels[i, self.terms_dict[t_id]] = 1
        data = [data_gos, data_exp]
        return (data, labels)
    
    
if __name__ == '__main__':
    main()
