#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:32:43 2018

@author: lsm
"""

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.utils import expand_path, set_deeppavlov_root
from deeppavlov.core.common.file import read_json,save_json
import numpy as np
import pandas as pd
from deeppavlov.core.commands.infer import *
from model.pipeline.embedder import *
from model.pipeline.CNN_model import *
from model.pipeline.text_normalizer import *
from utils.stop_words_remover import *
from utils.data_equalizer import DataEqualizer
from utils.embeddings_builder import EmbeddingsBuilder
from utils.check_config import check_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import gc
import os
import pickle
from shutil import copy

class InvalidDataFormatError(Exception):
    pass

class InvalidDataAugmentationMethodError(Exception):
    pass

class InvalidModelLevelError(Exception):
    pass

class InvalidConfig(Exception):
    def __init__(self,errors,mes):
        save_json(errors,'config_chceck_report.json')
class IntentsClassifier():
    def __init__(self, root_config_path, 
                 sub_configs = {
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
        
        #root_config = read_json(root_config_path)
        #self.__root_model = build_model_from_config(root_config)
        
        
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
            'confidence': np.sort(res)[0,-len(dec):].tolist()[::-1]
               }
    
    def train(self, model_level, model_name, path_to_data, path_to_config, path_to_global_embeddings,
              test_size = 0.15, aug_method = 'word_dropout', samples_per_class = None,
              class_names = None,
              path_to_save_file = None,
              path_to_resulting_file = None):
        #preparing training/testing data
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
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        if not os.path.isdir(model_path+'data/'):
            os.mkdir(model_path+'data/')
        df_train_equalized.to_csv(model_path+'data/train.csv')
        df_val[['text', 'labels']].sample(frac = 1).to_csv(model_path+'data/valid.csv')
        df_test[['text', 'labels']].sample(frac = 1).to_csv(model_path+'df_test.csv')
        
        #preparing model
        config = read_json(path_to_config)
        
        #making embeddings
        eb = EmbeddingsBuilder(resulting_dim=config['chainer']['pipe'][2]['emb_len'],
                               path_to_original_embeddings=path_to_global_embeddings)
        tc = TextCorrector()
        corpus_cleaned = tc.tn.transform(df_raw.text.tolist())
        if not os.path.isfile(model_path+'ft_compressed.pkl'):
            eb.compress_embeddings(corpus_cleaned,model_path+'ft_compressed.pkl','pca',eb.path_to_original_embeddings)
        gc.collect()
        if not os.path.isfile(model_path+'ft_compressed_local.pkl'):
            eb.build_local_embeddings(corpus_cleaned,model_path+'ft_compressed_local.pkl')
        #dealing with class_names
        if type(class_names)==list:
            pickle.dump(class_names, open(model_path+'class_names.pkl','wb'))
        else:
            pickle.dump(df_train['labels'].value_counts().index.tolist(), open(model_path+'class_names.pkl','wb'))
        #setting up saving and loading
        if not path_to_save_file == None:
            config['chainer']['pipe'][-1]['save_path'] = path_to_save_file+'weights.hdf5'
            save_json(config,path_to_config)
        if not os.path.isdir(path_to_save_file) and not path_to_save_file==None:
            os.mkdir(path_to_save_file)
            save_json(config,path_to_config)
            
        if not os.path.isdir(path_to_resulting_file) and not path_to_resulting_file == None:
            os.mkdir(path_to_resulting_file)
            save_json(config,path_to_config)

        check_results = check_config(path_to_config)
        if len(check_results)>0:
            raise InvalidConfig(check_results,'Config file is invalid')

        #training
        set_deeppavlov_root(config)
        #update training status
        training_status = 'Classification model {} {} is currently training. Total number of epochs is set to {}'.format(model_level, model_name, config['train']['epochs'])
        with open(model_path+'status.txt','w') as f:
            f.writelines(training_status)
        #fukken training
        train_evaluate_model_from_config(path_to_config)
        #updating status
        training_status = 'Classification model {} {} is trained'.format(model_level, model_name)
        with open(model_path+'status.txt','w') as f:
            f.writelines(training_status)
        #getting performance
        perf = self.get_performance(path_to_config, model_path+'df_test.csv')
        print('f1_macro: {}'.format(perf))
        copy(path_to_save_file+'weights.hdf5', path_to_resulting_file)
        
    def get_status(model_directory):
        with open(model_directory+'status.txt') as f:
            status = f.readlines()
        return status
        
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
    ic = IntentsClassifier(root_config_path='root/cf_config_dual_bilstm_cnn_model.json',sub_configs = sub_configs)
    ic.train('root','','df_raw.csv','root/cf_config_dual_bilstm_cnn_model.json', 
             path_to_global_embeddings = '/home/lsm/projects/general_purpose/embeddings/fasttext/ft_native_300_ru_wiki_lenta_lemmatize.bin',
             samples_per_class = 1500,
             class_names = ['доставка', 'оплата', 'другое','намерение сделать заказ'],
             path_to_save_file='temp_save/',
             path_to_resulting_file='temp/')
    mes = ''
    while mes != 'q':
        ic = IntentsClassifier(root_config_path='root/cf_config_dual_bilstm_cnn_model.json',sub_configs = sub_configs)
        mes = input()
        print(ic.run(mes))