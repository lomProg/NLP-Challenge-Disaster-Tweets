import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.spatial.distance import euclidean

from classification import DataGenerator as dg

class GloVe(object):

    MAX_SEQUENCE_LENGTH = 20

    def __init__(self, path:str) -> None:
        self.__glove_embeddings = self.__load_glove__(path)
        self.EMBEDDING_DIM = self.__glove_embeddings.get(b'a').shape[0]

    @staticmethod
    def __load_glove__(path:str)->dict:
        """ Loads Glove word embeddings in memory from the file stored
        in `path`. """
        embeddings_dict={}
        with open(path,'rb') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        return embeddings_dict

    def find_closest_embeddings(self,
                                tgt_embedding:np.ndarray,
                                n_words:int=5)->list:
        """Generates an ordered list of the `n_words` words in the
        embeddings dictionary most similar to the target word
        `tgt_embedding` given as input.

        Parameters
        ----------
        tgt_embedding : np.ndarray
            Target word from which to find the most similar words
        n_words : int, optional
            Desired number of similar words to be returned, by default 5

        Returns
        -------
        list
            It collects the `n_words` words most similar to the target
            word, ordered by their Euclidean distance from
            `tgt_embedding`.
        """
        sorted_embeddings = sorted(self.__glove_embeddings.keys(),
                                   key=lambda word:
                                   euclidean(self.__glove_embeddings[word],
                                             tgt_embedding))
        return sorted_embeddings[:n_words]
    
    def __create_matrix__(self)->np.ndarray:
        """Starting from the stored vocabulary, it defines and populates
        the embedding matrix where each entry corresponds to a
        vocabulary token.

        Returns
        -------
        np.ndarray
            The embedding matrix has shape `(vocabulary length,
            embedding length)`, where the embedding vector length
            corresponds to the value stored in `self.EMBEDDING_DIM`. 
        """
        num_words = len(self.vocabulary) + 1
        matrix = np.zeros((num_words, self.EMBEDDING_DIM))
        for i, word in self.vocabulary.items():
            matrix[i] = self.__glove_embeddings.get(
                word.encode("utf-8"),
                np.zeros(self.EMBEDDING_DIM))
        return matrix

    def prepare_data(self,
                     x:pd.Series,
                     y:pd.Series,
                     **kwargs):

        gen = dg()
        # Splitting input data
        gen.split_data(x, y, **kwargs)
        # Data tokenization
        gen.tokenize_data(max_sequence_length=self.MAX_SEQUENCE_LENGTH)

        self.data = gen.data
        self.vocabulary = gen.vocabulary

        embedding_matrix = self.__create_matrix__()
        self.embedding_matrix = embedding_matrix

########################################################################
# Word2Vec
class W2V(object):


    def __init__(self, model_name:str) -> None: #costruttore
        self.name = model_name

    def build_model(
            self, #chiamata a me stesso
            x:pd.Series,
            y:pd.Series,
            #feature:str,
            model_path:str,
            #model_name:str,
            vector_size:int=200,
            window:int=2,            
            save_model = True,
            **kwargs) -> None:
        """ It builds, trains and saves a Word2vec model based on the
        parameter in input.

        Parameters
        ----------
        vector_size : int
            Length of the embedding vector
        window : int
            Context window size
        train_set : pd.DataFrame       
            Dataset on which to train the Word2Vec model
        feature : str
            Feature of the dataset to vectorize e.g. 'tokenized_text'
        model_path : str        
            Path where to save the model
        """
        gen = dg()
        # Splitting input data
        gen.split_data(x, y, **kwargs)
        self.data = gen.data

        data = gen.data['x_train']

        model_w2v = Word2Vec(
            data,
            vector_size = vector_size,
            window = window,
            min_count = 3, # Ignores all words with total frequency lower than 3                               
            sg = 0, # 1 for skip-gram model, 0 for CBOW model
            hs = 0,
            seed = 34,
            min_alpha = 0.05
        ) 

        model_w2v.train(data, total_examples = len(data), epochs = 25)
        self.model = model_w2v

        words = list(model_w2v.wv.index_to_key)
        self.vocabulary = words
        print('Number of words in the vocabulary:')
        print(len(words))

    def write_w2v_csv(
            self,
            data:pd.DataFrame,
            model, path:str,
            feature:str,
            vector_size:int=200):
        """ It creates and saves a pd.DataFrame() containing the embedding
        vector of each tokenized document present in the Dataframe in input.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame in input
        model : _type_
            A gensim.models.Word2vec document already trained
        path : str
            Path where to save the .csv file generated by the function
        feature : str        
            Column of the DataFrame in input to vectorize
            e.g. 'tokenized_text'
        vector_size : int, optional
            Number of columns of the embedding dataset, by default 200
        """
    # Store the vectors for data in .csv file saved in a specific path
        words = self.vocabulary

        with open(path, 'w+') as word2vec_file:
            for index, row in data.iterrows():
                # Text embedding computed as the mean of the embedding
                # vector of each word in the document
                model_vector = (np.mean([model.wv[token] for token in row[feature] if token in words],
                                        axis=0)).tolist()
                if index == 0:
                    header = ",".join(str(ele) for ele in range(vector_size))
                    word2vec_file.write(header)
                    word2vec_file.write("\n")
                # Check if the line exists else it is vector of zeros
                if type(model_vector) is list:  
                    line1 = ",".join( [str(vector_element) for vector_element in model_vector] )
                else:
                    line1 = ",".join([str(0) for i in range(vector_size)])
                word2vec_file.write(line1)
                word2vec_file.write('\n')
                