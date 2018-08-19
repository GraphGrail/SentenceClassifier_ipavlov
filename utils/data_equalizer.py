import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import random


class DataEqualizer():
    
    def __init__(self):
        pass
    
    def __augment_sentence(self, sent,n_augmentations = 1):
        tokens = word_tokenize(sent)
        res = []
        for n in range(n_augmentations):
            toremove = random.choice(tokens)
            res.append(' '.join([t for t in tokens if t not in [toremove]]))
        return res
    
    def equalize_classes(self, df_train, samples_per_class, aug_method):
        all_classes = df_train['labels'].value_counts().index.tolist()
        all_classes_n = df_train['labels'].value_counts().tolist()
        texts = []
        labels = []
        for c, cn in zip(all_classes, all_classes_n):
            if samples_per_class == None:
                tdf = df_train[df_train['labels'] == c]
            elif cn<samples_per_class:
                tofill = samples_per_class-cn
                class_df = df_train[df_train['labels'] == c]
                filling_idx = np.random.randint(low = 0, high = class_df.shape[1],size=tofill)
                if aug_method == 'duplicate':
                    tdf = class_df.iloc[filling_idx,:]
                    tdf = pd.concat([tdf,class_df])
                elif aug_method == 'word_dropout':
                    tdf = class_df.iloc[filling_idx,:]
                    tdf['text'] = tdf['text'].apply(self.__augment_sentence)
                    tdf = pd.concat([tdf,class_df])
            elif cn>=samples_per_class:
                class_df = df_train[df_train['labels'] == c]
                tdf = class_df.sample(samples_per_class)
            
            texts.append(tdf.text.values)
            labels.append(tdf['labels'].values)
        res_x = np.hstack(texts)
        res_y = np.hstack(labels)
        return pd.DataFrame(data=np.vstack([res_x,res_y]).T,columns = ['text', 'labels'])
