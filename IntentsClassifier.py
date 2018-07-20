#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:32:43 2018

@author: lsm
"""

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.utils import expand_path, set_deeppavlov_root
from deeppavlov.core.common.file import read_json
import numpy as np
import pandas as pd
from deeppavlov.core.commands.infer import *
from model.pipeline.embedder import *
from model.pipeline.CNN_model import *
from model.pipeline.text_normalizer import *
from utils.data_equalizer import DataEqualizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class InvalidDataFormatError(Exception):
    pass

class InvalidDataAugmentationMethodError(Exception):
    pass

class InvalidModelLevelError(Exception):
    pass

class IntentsClassifier():
    def __init__(self, root_config_path, 
                 sub_configs = {
            'оплата':'subs/pay/cf_config_dual_bilstm_cnn_model.json',
            'доставка': 'subs/deliver/cf_config_dual_bilstm_cnn_model.json'
            }):
        self.__root_config = root_config_path
        self.__sub_models = {}
        if len(list(sub_configs.items()))>0:
            self.__sub_configs = sub_configs
            for cl,conf in self.__sub_configs.items():
                sc = read_json(conf)
                self.__sub_models[cl] = build_model_from_config(sc)
        else:
            self.__sub_configs = {'not present':''}
        
        root_config = read_json(root_config_path)
        self.__root_model = build_model_from_config(root_config)
        
        
        self.__data_equalizer = DataEqualizer()

    
    def __predict(self, model, input_text):
        rp = model.pipe[0][-1]([input_text])
        for i in range(1,len(model.pipe)-1):
            rp = model.pipe[i][-1](rp)
        res = model.pipe[-1][-1](rp, predict_proba = True)
        dec = proba2labels(res, 
                           confident_threshold = model.pipe[-1][-1].confident_threshold,
                           classes=model.pipe[-1][-1].classes)[0]
        return {
            'decision': dec,
            'confidence': np.max(res)
               }
    
    def train(self, model_level, model_name, path_to_data, path_to_config, 
              test_size = 0.15, aug_method = 'word_dropout', samples_per_class = None):
        df_raw = pd.read_csv(path_to_data)
        
        if 'labels' not in df_raw or 'text' not in df_raw:
            raise InvalidDataFormatError('\'labels\' and \'text\' columns must be in the dataframe')
        
        if model_level not in ['root', 'subs']:
            raise InvalidModelLevelError('model level should be either \'root\' or \'subs\'')
        
        __df_train, df_test, _, _ = train_test_split(df_raw, df_raw, test_size=test_size)
        df_train, df_val, _, _ = train_test_split(__df_train, __df_train, test_size=test_size)
        
        if aug_method not in ['word_dropout', 'duplicate']:
            raise InvalidDataAugmentationMethodError('\'aug_method\' should be  \'word_dropout\' or \'duplicate\'')
        
        df_train_equalized = self.__data_equalizer.equalize_classes(df_train, samples_per_class, aug_method)

        model_path = model_level+ '/'
        if model_level == 'subs':
            model_path += model_name + '/'
            
        df_train_equalized.to_csv(model_path+'data/train.csv')
        df_val[['text', 'labels']].sample(frac = 1).to_csv(model_path+'data/valid.csv')
        df_test[['text', 'labels']].sample(frac = 1).to_csv(model_path+'df_test.csv')
        
        config = read_json(path_to_config)
        set_deeppavlov_root(config)
        train_evaluate_model_from_config(path_to_config)
        perf = self.get_performance(path_to_config, model_path+'df_test.csv')
        print('f1_macro: {}'.format(perf))
        
        
    def get_performance(self, path_to_config, path_to_test_data):    
        df_test = pd.read_csv(path_to_test_data)
        if 'labels' not in df_test or 'text' not in df_test:
            raise InvalidDataFormatError('\'labels\' and \'text\' columns must be in the dataframe')
        config = read_json(path_to_config)
        model = build_model_from_config(config)
        def eval_ipavlov(in_x):
            in_s = []
            in_s.append('{}::'.format(in_x))
            return model(in_s)[0][0]
        df_test['labels'] = df_test['labels'].apply(lambda x: x.lower())
        preds = df_test.text.apply(eval_ipavlov)
        f1_macro = f1_score(preds, df_test['labels'], average = 'macro')
        return {'f1_macro':f1_macro}
        
        
        
    def run(self,message):
        res = {}
        root_config = read_json(self.__root_config)
        root_model = build_model_from_config(root_config)
        root_res = self.__predict(root_model,message)
        res['root'] = root_res
        res['subs'] = {}
        for dec in root_res['decision']:
            if dec in list(self.__sub_configs.keys()):
                sc = read_json(self.__sub_configs[dec])
                sub_model = build_model_from_config(sc)
                res['subs'][dec] = self.__predict(sub_model,message)
        return res
    
if __name__ == '__main__':
    sub_configs = {
            'оплата':'subs/pay/cf_config_dual_bilstm_cnn_model.json',
            'доставка': 'subs/deliver/cf_config_dual_bilstm_cnn_model.json'
            }
    
    mes = ''
    while mes != 'q':
        ic = IntentsClassifier(root_config_path='root/cf_config_dual_bilstm_cnn_model.json',sub_configs = {})
        mes = input()
        print(ic.run(mes))
    ic.train('subs','pay','df_raw.csv','subs/pay/cf_config_dual_bilstm_cnn_model.json', samples_per_class = 1500)