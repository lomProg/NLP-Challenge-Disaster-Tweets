import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.spatial.distance import euclidean

import inspect
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization

from classification import DataGenerator as dg

__all__ = ["WordEmbedding", "GloVe", "W2V"]

class WordEmbedding:


    def __init__(self,
                 model_name:str,
                 model_path:str=None) -> None:
        self.name = model_name
        self.path = model_path

    def __str__(self):
        if self.path:
            return f"Model {self.name} stored at {self.path}"
        else:
            return f"Model {self.name}"


class GloVe(WordEmbedding):

    MAX_SEQUENCE_LENGTH = 20

    def __init__(self,
                 model_path:str,
                 model_name:str="model") -> None:
        super().__init__(model_name, model_path)
        self.glove_embeddings = self.__load_glove__()
        self.EMBEDDING_DIM = self.glove_embeddings.get(b'a').shape[0]

    def __load_glove__(self) -> dict:
        """ Loads Glove word embeddings in memory from the file stored
        in `self.path`. """
        embeddings_dict={}
        with open(self.path,'rb') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        return embeddings_dict

    def __create_matrix__(self) -> np.ndarray:
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
            matrix[i] = self.glove_embeddings.get(
                word.encode("utf-8"),
                np.zeros(self.EMBEDDING_DIM))
        return matrix

    def find_closest_embeddings(self,
                                tgt_embedding:np.ndarray,
                                n_words:int=5) -> list:
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
        sorted_embeddings = sorted(self.glove_embeddings.keys(),
                                   key=lambda word:
                                   euclidean(self.glove_embeddings[word],
                                             tgt_embedding))
        return sorted_embeddings[:n_words]

    def prepare_data(self,
                     x:pd.Series,
                     y:pd.Series,
                     **kwargs) -> None:

        splitting_args = list(inspect.signature(train_test_split).parameters)
        splitting_dict = {k: kwargs.pop(k)
                          for k in dict(kwargs) if k in splitting_args}
        token_args = list(inspect.signature(TextVectorization).parameters)
        token_args = list(filter(lambda x: x not in ["output_sequence_length"],
                                 token_args))
        token_dict = {k: kwargs.pop(k)
                      for k in dict(kwargs) if k in token_args}
        gen = dg(x, y)
        if (("test_size" not in splitting_dict and
             "train_size" not in splitting_dict) or
             ("test_size" in splitting_dict and
              splitting_dict["test_size"] == 0) or
              ("train_size" in splitting_dict and
               splitting_dict["train_size"] == 1)):
            # If splitting of data into train and test is not required,
            # the split value for train or test will equal 1 or 0
            # respectively.
            gen.tokenize_data(max_sequence_length=self.MAX_SEQUENCE_LENGTH,
                              **token_dict)
        else:
            # Splitting input data
            gen.split_data(**splitting_dict)
            # Data tokenization
            gen.tokenize_data(max_sequence_length=self.MAX_SEQUENCE_LENGTH,
                              **token_dict)

        self.data = gen.data
        self.vocabulary = gen.vocabulary

        embedding_matrix = self.__create_matrix__()
        self.embedding_matrix = embedding_matrix


class W2V(WordEmbedding):

    MAX_SEQUENCE_LENGTH = 20

    # Constructor
    def __init__(self,
                 model_name:str,
                 model_path:str=None) -> None:
        super().__init__(model_name, model_path)

    def __create_matrix__(self) -> np.ndarray:
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
        if not hasattr(self, 'vocabulary'):
            self.vocabulary = list(self.model.wv.index_to_key)
        num_words = len(self.model.wv.index_to_key) + 1
        matrix = np.zeros((num_words, self.EMBEDDING_DIM))
        for word, i in self.vocabulary.items():
            matrix[i] = self.model.wv[word] 
        return matrix

    @classmethod
    def load_model(cls,
                   x:pd.Series,
                   y:pd.Series,
                   model_name:str,
                   model_path:str):
        

        w2v_obj = cls(model_name)
        w2v_obj.path = model_path
        w2v_obj.model = Word2Vec.load(w2v_obj.path)
        return w2v_obj

    def build_model(self,
                    x:pd.Series,
                    y:pd.Series,
                    vector_size:int=200,
                    save_model = True,
                    dst_model_path=None,
                    **kwargs) -> None:
        """It builds, trains and saves a Word2vec model based on the
        parameters in input.

        Parameters
        ----------
        x : pd.Series
            Data on which to train the Word2Vec model
        y : pd.Series
            Target data
        vector_size : int, optional
            Length of the embedding vector, by default 200
        save_model : bool, optional
            If True the trained model will be saved, by default True
        dst_model_path : _type_, optional
            Path where to save the model, by default None

        Raises
        ------
        AttributeError
            If no destination path is specified for saving the model.
        """
        self.EMBEDDING_DIM = vector_size
        # splitting_args = list(inspect.signature(train_test_split).parameters)
        # splitting_dict = {k: kwargs.pop(k)
        #                  for k in dict(kwargs) if k in splitting_args}
        
        token_args = list(inspect.signature(TextVectorization).parameters)
        token_args = list(filter(lambda x: x not in ["output_sequence_length"],
                                 token_args))
        token_dict = {k: kwargs.pop(k)
                      for k in dict(kwargs) if k in token_args}
        gen = dg(x, y)
        gen.tokenize_data(max_sequence_length=self.MAX_SEQUENCE_LENGTH,
                              **token_dict)
        data = gen.data.copy()
        # Re-building the tweets from the vocabulary
        data['token_x'] = pd.Series([[gen.vocabulary[i]
                                      for i in gen.data['vect_x'].iloc[k]
                                      if len(gen.vocabulary[i]) != 0]
                                      for k in range(len(gen.data['vect_x']))],
                                      index=gen.data['x'].index)
        self.data = data
        # Inspecting parameters
        w2v_args = list(inspect.signature(Word2Vec).parameters)
        w2v_args = list(filter(lambda x: x not in ["min_alpha", "vector_size"],
                               w2v_args))
        w2v_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in w2v_args}
        # Build model
        model_w2v = Word2Vec(
            data['token_x'],
            min_count = 0, ####
            vector_size = self.EMBEDDING_DIM,
            min_alpha = 0.05,
            **w2v_dict
            )

        model_w2v.train(data['token_x'], total_examples=len(data['token_x']),
                        epochs=25)
        self.model = model_w2v

        words = list(model_w2v.wv.index_to_key)
        self.vocabulary = words
        print(f'Number of words in the vocabulary:\t{len(words)}')

        # Saving model
        if save_model:
            if hasattr(self, "path") or dst_model_path:
                word2vec_model_path = self.path + self.name + '.model'
                model_w2v.save(word2vec_model_path)
                print("The word2vec model has been trained and saved")
            else:
                err_msg = "No destination path in which to save the model"
                raise AttributeError(err_msg)
            
    def prepare_data(self,
                     nn_classifier:bool=True):
        # Split train e test
        if nn_classifier:
            if hasattr(self, 'data'):
                embedding_matrix = self.__create_matrix__()
            #else: 

            self.embedding_matrix = embedding_matrix

    def vectorization(self,
                      data,
                      save_dataframe:bool=True,
                      dst_df_path:str="./",
                      dst_df_name:str="embedding_data") -> pd.Series:
        """ It creates a pd.Series and optionally saves it as a pd.DataFrame()
        containing the embedding vector of each tokenized text present in
        the Dataframe in input.

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
        vectors = []
        for row in data:
            vec = (np.mean([self.model.wv[token]
                            if token in self.vocabulary
                            else np.array([0]*self.EMBEDDING_DIM)
                            for token in row], axis = 0))
            vectors.append(vec)

        if save_dataframe:
            df = pd.DataFrame({'vector':pd.Series(vectors)})
            df_path = dst_df_path + dst_df_name + '.csv'
            df.to_csv(df_path, header = True, index = False)

        return pd.Series(vectors)