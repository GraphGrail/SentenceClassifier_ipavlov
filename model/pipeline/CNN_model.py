#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:54:05 2018

@author: lsm
"""
from typing import Dict
from deeppavlov.core.models.keras_model import KerasModel
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.registry import register
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Input, concatenate,Activation,Concatenate,Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.core import Dropout
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.layers import Input, Dense, Embedding,Concatenate
from deeppavlov.models.classifiers.utils import labels2onehot, log_metrics, proba2labels
from deeppavlov.core.common.log import get_logger
import numpy as np
import pickle
from keras import regularizers

log = get_logger(__name__)

@register('cnn_model')
class CNN_classifier(KerasModel):
    def __init__(self,
                 save_path,
                 load_path,
                 architecture_name: str,
                 architecture_params: Dict,
                 loss: str = 'categorical_crossentropy',
                 metrics: str = 'categorical_accuracy',
                 optimizer: str = 'adam',
                 confident_threshold=0.5,
                 classes='class_names.pkl',
                 **kwargs):
        opt = {}
        self.confident_threshold = confident_threshold
        self.classes = pickle.load(open(classes, 'rb'))
        self.n_classes = len(self.classes)
        self.save_path = save_path
        self.load_path = load_path
        architectures = {
            'cnn': self.cnn_model,
            'dual_bilstm_cnn_model': self.dual_bilstm_cnn_model
        }
        if architecture_name == 'cnn':
            params_list = [
                'cnn_layers',
                'emb_dim',
                'seq_len',
                'pool_size',
                'dropout_power',
            ]
        elif architecture_name == 'dual_bilstm_cnn_model':
            params_list = [
                'bilstm_layers',
                'conv_layers',
                'emb_dim',
                'seq_len',
                'pool_size',
                'dropout_power'
            ]
        else:
            raise NotImplementedError()
        for param in params_list:
            if param in architecture_params:
                opt[param] = architecture_params[param]
            else:
                raise NameError()

        model_builder = architectures[architecture_name]
        self.model = model_builder(opt)
        self.model.compile(optimizer=optimizer,
                           metrics=metrics,
                           loss=loss,
                           )
        if kwargs['mode'] == 'infer':
            self.load()

    def __call__(self, data, predict_proba=False, *args):
        """
        Infer on the given data
        Args:
            data: [list of sentences]
            predict_proba: whether to return probabilities distribution or only labels-predictions
            *args:

        Returns:
            for each sentence:
                vector of probabilities to belong with each class
                or list of labels sentence belongs with
        """
        preds = np.vstack([self.infer_on_batch(d) for d in data])

        if predict_proba:
            return preds
        else:
            pl = proba2labels(preds, confident_threshold=self.confident_threshold, classes=self.classes)
            tl = proba2labels(preds, confident_threshold=0,
                              classes=self.classes)  # [self.classes[np.argmax(preds[i,:])-1] for i in range(len(self.classes))]
            return [(pl[i], dict(zip(tl[i], preds[i]))) for i in range(len(pl))]  # [[y[0]] for y in pl]

    def train_on_batch(self, xa, ya):
        def add_noise(feats, labels, num_noise):
            fn = feats
            ln = labels
            for i in range(num_noise):
                noise = np.random.normal(1, 0.02, feats.shape)
                noised = feats * noise
                fn = np.vstack([fn, noised])
                ln = np.vstack([ln, labels])
            return fn, ln

        feats = list(map(list, zip(*list(xa))))
        labels = labels2onehot(np.array(ya), classes=self.classes)
        # va, la = add_noise(vectors, labels, 10)
        netin = [np.vstack(f) for f in feats]
        metrics_values = self.model.train_on_batch(netin, labels)
        return metrics_values[0]

    def infer_on_batch(self, batch, labels=None):
        if labels:
            onehot_labels = labels2onehot(labels, classes=np.arange(1, 20))
            metrics_values = self.model.test_on_batch(batch, onehot_labels)
            return metrics_values
        else:
            predictions = self.model.predict(batch)
            return predictions

    def cnn_model(self, opt):
        cnn_layers = opt['cnn_layers']
        emb_dim = opt['emb_dim']
        seq_len = opt['seq_len']
        pool_size = opt['pool_size']
        dropout_power = opt['dropout_power']

        model = Sequential()
        model.add(Conv1D(filters=cnn_layers[0]['filters'],
                         kernel_size=cnn_layers[0]['kernel_size'],
                         input_shape=(seq_len, emb_dim)))

        for i in range(1, len(cnn_layers)):
            model.add(Conv1D(filters=cnn_layers[i]['filters'],
                             kernel_size=cnn_layers[i]['kernel_size']))

        model.add(MaxPooling1D(pool_size))
        model.add(Flatten())
        model.add(Dropout(dropout_power))
        model.add(Dense(self.n_classes, activation='softmax'))

        return model

    def dual_bilstm_cnn_model(self, opt):
        bilstm_layers = opt['bilstm_layers']
        conv_layers = opt['conv_layers']
        emb_dim = opt['emb_dim']
        seq_len = opt['seq_len']
        pool_size = opt['pool_size']
        dropout_power = opt['dropout_power']
        # Left wing
        li0 = Input(shape=(seq_len, emb_dim))
        lr = li0
        for bilstm_layer in bilstm_layers:
            lr = Bidirectional(layer=LSTM(units=bilstm_layer['units'],
                                          activation=bilstm_layer['activation'],
                                          kernel_regularizer=regularizers.l2(bilstm_layer['l2_power']),
                                          return_sequences=True))(lr)
        lconv = lr
        for conv_layer in conv_layers:
            lconv = Conv1D(filters=conv_layer['units'],
                           kernel_size=conv_layer['kernel_size'],
                           activation=conv_layer['activation'],
                           kernel_regularizer=regularizers.l2(conv_layer['l2_power']),
                           kernel_initializer='he_normal')(lconv)
        lbn = BatchNormalization()(lconv)
        lmp = MaxPooling1D(pool_size)(lbn)
        lf = Flatten()(lmp)
        ldr = Dropout(dropout_power)(lf)

        # Right wing
        ri0 = Input(shape=(seq_len, emb_dim))
        rr = ri0
        for bilstm_layer in bilstm_layers:
            rr = Bidirectional(layer=LSTM(units=bilstm_layer['units'],
                                          activation=bilstm_layer['activation'],
                                          kernel_regularizer=regularizers.l2(bilstm_layer['l2_power']),
                                          return_sequences=True))(rr)
        rconv = rr
        for conv_layer in conv_layers:
            rconv = Conv1D(filters=conv_layer['units'],
                           kernel_size=conv_layer['kernel_size'],
                           activation=conv_layer['activation'],
                           kernel_regularizer=regularizers.l2(conv_layer['l2_power']),
                           kernel_initializer='he_normal')(rconv)
        rbn = BatchNormalization()(rconv)
        rmp = MaxPooling1D(pool_size)(rbn)
        rf = Flatten()(rmp)
        rdr = Dropout(dropout_power)(rf)

        un = Concatenate()([ldr, rdr])
        out = Dense(self.n_classes, activation='softmax')(un)

        net = Model(inputs=[li0, ri0], outputs=out)

        return net

    def load(self):
        self.model.load_weights(self.load_path)

    def save(self):
        self.model.save_weights(self.save_path)

    def reset(self):
        pass
    