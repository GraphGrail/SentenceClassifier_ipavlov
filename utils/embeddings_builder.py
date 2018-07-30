import fastText
import pickle
from sklearn.decomposition import PCA
import numpy as np
from nltk import word_tokenize
import gc
import shutil
import os

class InvalidDataFormatError(Exception):
    pass

class EmbeddingsBuilder():
    
    def __init__(self, resulting_dim, path_to_original_embeddings):
        self.path_to_original_embeddings = path_to_original_embeddings
        self.resulting_dim = resulting_dim
        
    def build_local_embeddings(self,corpus, path_to_resulting_embeddings):
        if not os.path.isdir('temp/'):
            os.mkdir('temp/')
        with open('temp/corpus.txt','w',encoding='utf-8',) as f:
            f.writelines(corpus)
        ft = fastText.train_unsupervised('temp/corpus.txt',minCount=1)
        ft.save_model('temp/ft.bin')
        del ft
        self.compress_embeddings(corpus, path_to_resulting_embeddings, 'pca', 'temp/ft.bin')
        shutil.rmtree('temp/')
        
    def compress_embeddings(self, unified_corpus, path_to_resulting_embeddings,method, 
                            path_to_original_embeddings):
        try:
           #ft = FastText.load(self.path_to_original_embeddings)
           ft = fastText.load_model(path_to_original_embeddings)
        except InvalidDataFormatError:
            print('unable to load fasttext format')
        if method == 'pca':
            unique_words = list(set([w for w in word_tokenize(' '.join(unified_corpus))]))#ft.get_words()
            vecs = np.vstack([ft.get_word_vector(w) for w in unique_words])
            pca = PCA(n_components=self.resulting_dim).fit(vecs)
            vecs_pca = pca.transform(vecs)
            compressed = dict(zip(unique_words,vecs_pca))
            pickle.dump(compressed, open(path_to_resulting_embeddings,'wb'))
        elif method == 'quantize':
            raise NotImplementedError()
        
if __name__ == '__main__':
    eb = EmbeddingsBuilder(resulting_dim=50,
                           path_to_original_embeddings='../../../../../../general_purpose/embeddings/fasttext/ft_native_300_ru_wiki_lenta_lemmatize.bin')
    eb.compress_embeddings('привет','res.pkl','pca',eb.path_to_original_embeddings)
    gc.collect()
    eb.build_local_embeddings(['привет'],'ft_local.pkl')