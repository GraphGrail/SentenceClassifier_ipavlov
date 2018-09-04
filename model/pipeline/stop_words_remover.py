from overrides import overrides

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.serializable import Serializable
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

log = get_logger(__name__)

@register('stop_words_remover')
class StopWordsRemover(Component, Serializable):
    
    def __cleanupDoc(self, s,stopset):
        tokens = word_tokenize(s)
        cleanup = [token.lower() for token in tokens if token not in stopset and  len(token)>2]
        return ' '.join(cleanup)
    
    def __init__(self, language = 'rus', **kwargs):
        if not 'stopset' in kwargs:
            if language == 'rus':
                self.stopset = set(stopwords.words('russian'))
            elif language == 'en':
                self.stopset = set(stop_words.ENGLISH_STOP_WORDS)
            else:
                self.stopset = set([])
        else:
            self.stopset = kwargs['stopset']

    def save(self, *args, **kwargs):
        raise NotImplementedError
        
    def load(self, *args, **kwargs):
        pass

    @overrides
    def __call__(self, texts, *args, **kwargs):
        assert type(texts)==list
        return [self.__cleanupDoc(text,self.stopset) for text in texts]