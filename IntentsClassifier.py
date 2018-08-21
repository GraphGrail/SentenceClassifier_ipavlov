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
from deeppavlov.models.classifiers.utils import proba2labels
from SentenceClassifier_ipavlov.model.pipeline.embedder import *
from SentenceClassifier_ipavlov.model.pipeline.CNN_model import *
from SentenceClassifier_ipavlov.model.pipeline.text_normalizer import *
from SentenceClassifier_ipavlov.utils.stop_words_remover import *
from SentenceClassifier_ipavlov.utils.data_equalizer import DataEqualizer
from SentenceClassifier_ipavlov.utils.embeddings_builder import EmbeddingsBuilder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from deeppavlov.core.common.metrics_registry import get_metrics_by_names
from deeppavlov.core.common.registry import get_model

import gc
import os
import pickle
from shutil import copy
import json

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
        def prepare_config(config):
            model_path = config['model_path']
            emb_config = IntentsClassifier.get_config_element_by_name(config['chainer']['pipe'], 'embedder')
            cnn_config = IntentsClassifier.get_config_element_by_name(config['chainer']['pipe'], 'cnn_model')
            config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][0] = os.path.abspath(
                model_path) + '/' + config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][
                                                                                                         0]
            config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][1] = os.path.abspath(
                model_path) + '/' + config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][
                                                                                                         1]
            config['chainer']['pipe'][config['chainer']['pipe'].index(cnn_config)]['classes'] = os.path.abspath(
                model_path) + '/' + config['chainer']['pipe'][config['chainer']['pipe'].index(cnn_config)]['classes']
            config['dataset_reader']['data_path'] = os.path.abspath(model_path) + '/' + config['dataset_reader'][
                'data_path']
            config['chainer']['pipe'][-1]['load_path'] = os.path.abspath(model_path)+'/'+config['chainer']['pipe'][-1]['load_path']
            return config

        self.__root_config = root_config_path
        self.__sub_models = {}
        if len(list(sub_configs.items()))>0:
            self.__sub_configs = sub_configs
            for cl,conf in self.__sub_configs.items():
                sc = read_json(conf)
                sc = prepare_config(sc)
                self.__sub_models[cl] = build_model_from_config(sc)
        else:
            self.__sub_configs = {'not present':''}
        
        self.__data_equalizer = DataEqualizer()
    @classmethod
    def get_config_element_by_name(cls, config, name):
        els = [el for el in config if el['name']==name]
        if len(els)>0:
            return els[0]
        else:
            raise InvalidConfig(errors = None,mes = 'element %s is not found'%(name))
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

    @classmethod
    def check_config(cls, config):
        def check_metrics_registry(model_name):
            try:
                get_metrics_by_names(model_name)
            except BaseException:
                return False
            return True
        
        def check_model_registry(model_name):
            try:
                get_model(model_name)
            except BaseException:
                return False
            return True
        def check_file_existance(filepath):
            if not os.path.exists(filepath) and not filepath == "":
                return False
            return True
            
        def find_values(id, json_repr):
            results = []
        
            def _decode_dict(a_dict):
                try: results.append(a_dict[id])
                except KeyError: pass
                return a_dict
        
            json.loads(json_repr, object_hook=_decode_dict)  # Return value ignored.
            return results
        
        if type(config)==str:
            config = read_json(config)
        
        models = find_values('name',json.dumps(config))
        invalid_fields = []
        for model in models:
            if not check_model_registry(model):
                invalid_fields.append(model)
        emb = IntentsClassifier.get_config_element_by_name(config = config['chainer']['pipe'],name = 'embedder')
        if not check_file_existance(emb['load_path'][0]):
            invalid_fields.append(emb['load_path'][0])
        
        if not check_file_existance(emb['load_path'][1]):
            invalid_fields.append(emb['load_path'][1])
        
        mdl = IntentsClassifier.get_config_element_by_name(config = config['chainer']['pipe'],name = 'cnn_model')
        
        if not check_file_existance(mdl['classes']):
            invalid_fields.append(mdl['classes'])
        
        invalid_metrics = []
        
        for metric in config['train']['metrics']:
            if not check_metrics_registry([metric]):
                invalid_metrics.append(metric)
        if len(invalid_metrics)>0:
            invalid_fields += invalid_metrics
            
        return invalid_fields
    
    def train(self, model_level, model_name, path_to_data, path_to_config, path_to_global_embeddings,
              test_size = 0.15, aug_method = 'word_dropout', samples_per_class = None,
              class_names = None,
              path_to_save_file = None,
              path_to_resulting_file = None):
        #preparing training/testing data
        df_raw = pd.read_csv(path_to_data)

        #preparing config
        config = read_json(path_to_config)
        
        if 'labels' not in df_raw or 'text' not in df_raw:
            raise InvalidDataFormatError('\'labels\' and \'text\' columns must be in the dataframe')
        
        if model_level not in ['root', 'subs']:
            raise InvalidModelLevelError('model level should be either \'root\' or \'subs\'')
        
        __df_train, df_test, _, _ = train_test_split(df_raw, df_raw, test_size=test_size)
        df_train, df_val, _, _ = train_test_split(__df_train, __df_train, test_size=test_size)
        
        if aug_method not in ['word_dropout', 'duplicate']:
            raise InvalidDataAugmentationMethodError('\'aug_method\' should be  \'word_dropout\' or \'duplicate\'')
        
        df_train_equalized = self.__data_equalizer.equalize_classes(df_train, samples_per_class, aug_method)

        model_path = config['model_path']

        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        if not os.path.isdir(model_path+'data/'):
            os.mkdir(model_path+'data/')
        df_train_equalized.to_csv(model_path+'data/train.csv')
        df_val[['text', 'labels']].sample(frac = 1).to_csv(model_path+'data/valid.csv')
        df_test[['text', 'labels']].sample(frac = 1).to_csv(model_path+'df_test.csv')
        


        #making embeddings
        emb_len = IntentsClassifier.get_config_element_by_name(config = config['chainer']['pipe'],name = 'embedder')['emb_len']
        eb = EmbeddingsBuilder(resulting_dim=emb_len,
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
        if not os.path.isdir(path_to_save_file) and not path_to_save_file==None:
            os.mkdir(path_to_save_file)
            
        if not os.path.isdir(path_to_resulting_file) and not path_to_resulting_file == None:
            os.mkdir(path_to_resulting_file)

        emb_config = IntentsClassifier.get_config_element_by_name(config['chainer']['pipe'],'embedder')
        cnn_config= IntentsClassifier.get_config_element_by_name(config['chainer']['pipe'],'cnn_model')
        config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][0] = os.path.abspath(model_path)+'/'+config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][0]
        config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][1] = os.path.abspath(model_path)+'/'+config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][1]
        config['chainer']['pipe'][config['chainer']['pipe'].index(cnn_config)]['classes'] = os.path.abspath(model_path)+'/'+config['chainer']['pipe'][config['chainer']['pipe'].index(cnn_config)]['classes']
        config['dataset_reader']['data_path'] = os.path.abspath(model_path)+'/'+config['dataset_reader']['data_path']
        load_path_bckp = config['chainer']['pipe'][-1]['load_path']
        check_results = self.check_config(config)
        if len(check_results)>0:
            raise InvalidConfig(check_results,'Config file is invalid')

        #training
        set_deeppavlov_root(config)
        #update training status
        training_status = 'Classification model {} {} is currently training. Total number of epochs is set to {}'.format(model_level, model_name, config['train']['epochs'])
        with open(model_path+'status.txt','w') as f:
            f.writelines(training_status)
        #fukken training

        train_evaluate_model_from_config(config)
        #fixing load_path
        #updating status
        training_status = 'Classification model {} {} is trained'.format(model_level, model_name)
        with open(model_path+'status.txt','w') as f:
            f.writelines(training_status)
        #getting performance
        perf = self.get_performance(config, model_path+'df_test.csv')
        print('f1_macro: {}'.format(perf['f1_macro']))
        config['chainer']['pipe'][-1]['load_path'] = load_path_bckp
        copy(os.path.abspath(path_to_save_file)+'/'+'weights.hdf5', os.path.abspath(path_to_resulting_file)+'/'+config['chainer']['pipe'][-1]['load_path'])
        copy(os.path.abspath(path_to_save_file) + '/' + 'weights.hdf5',
            os.path.abspath(model_path) + '/' + config['chainer']['pipe'][-1]['load_path'])
        
    def get_status(model_directory):
        with open(model_directory+'status.txt') as f:
            status = f.readlines()
        return status
        
    def get_performance(self, config, path_to_test_data):
        df_test = pd.read_csv(path_to_test_data)
        if 'labels' not in df_test or 'text' not in df_test:
            raise InvalidDataFormatError('\'labels\' and \'text\' columns must be in the dataframe')
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
        def prepare_config(config):
            model_path = config['model_path']
            emb_config = IntentsClassifier.get_config_element_by_name(config['chainer']['pipe'], 'embedder')
            cnn_config = IntentsClassifier.get_config_element_by_name(config['chainer']['pipe'], 'cnn_model')
            config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][0] = os.path.abspath(
                model_path) + '/' + config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][
                                                                                                         0]
            config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][1] = os.path.abspath(
                model_path) + '/' + config['chainer']['pipe'][config['chainer']['pipe'].index(emb_config)]['load_path'][
                                                                                                         1]
            config['chainer']['pipe'][config['chainer']['pipe'].index(cnn_config)]['classes'] = os.path.abspath(
                model_path) + '/' + config['chainer']['pipe'][config['chainer']['pipe'].index(cnn_config)]['classes']
            config['dataset_reader']['data_path'] = os.path.abspath(model_path) + '/' + config['dataset_reader'][
                'data_path']
            config['chainer']['pipe'][-1]['load_path'] = os.path.abspath(model_path)+'/'+config['chainer']['pipe'][-1]['load_path']
            return config
        res = {}
        root_config = read_json(self.__root_config)
        root_config=prepare_config(root_config)
        root_model = build_model_from_config(root_config)
        root_res = self.__predict(root_model,message)
        res['root'] = root_res
        res['subs'] = {}
        
        for dec in root_res['decision']:
            if dec in list(self.__sub_configs.keys()):
                sc = read_json(self.__sub_configs[dec])
                sc = prepare_config(sc)
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
