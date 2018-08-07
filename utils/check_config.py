#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:39:40 2018

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
from utils.data_equalizer import DataEqualizer
from utils.embeddings_builder import EmbeddingsBuilder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import gc
import os
import pickle
from shutil import copy
from deeppavlov.core.common.registry import get_model
from deeppavlov.core.common.metrics_registry import get_metrics_by_names
import json

def check_config(path_to_config):
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
            #raise InvalidModelError('{} is not a valid model name'.format(model_name))
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
    
    config = read_json(path_to_config)
    
    models = find_values('name',json.dumps(config))
    invalid_fields = []
    for model in models:
        if not check_model_registry(model):
            invalid_fields.append(model)
    
    if not  check_file_existance(config['chainer']['pipe'][2]['load_path'][0]):
        invalid_fields.append(config['chainer']['pipe'][2]['load_path'][0])
    
    if not check_file_existance(config['chainer']['pipe'][2]['load_path'][1]):
        invalid_fields.append(config['chainer']['pipe'][2]['load_path'][1])
    
    if not check_file_existance(config['chainer']['pipe'][-1]['classes']):
        invalid_fields.append(config['chainer']['pipe'][-1]['classes'])
    
    invalid_metrics = []
    
    for metric in config['train']['metrics']:
        if not check_metrics_registry([metric]):
            invalid_metrics.append(metric)
    if len(invalid_metrics)>0:
        invalid_fields += invalid_metrics
        
    return invalid_fields