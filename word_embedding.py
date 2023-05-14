import numpy as np
import pandas as pd
import gensim
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

class GloVe(object):

    MAX_SEQUENCE_LENGTH = 20

    def __init__(self, path:str) -> None:
        self.__glove_embeddings = self.__load_glove__(path)
        self.EMBEDDING_DIM = self.__glove_embeddings.get(b'Z').shape[0]

    def __load_glove__(path:str)->dict:
        """ Loads Glove word embeddings in memory from the file stored in
        `path`. """
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
        """Generates an ordered list of the `n_words` words in the embeddings
        dictionary most similar to the target word `tgt_embedding` given as
        input.

        Parameters
        ----------
        tgt_embedding : np.ndarray
            Target word from which to find the most similar words
        n_words : int, optional
            Desired number of similar words to be returned, by default 5

        Returns
        -------
        list
            It collects the `n_words` words most similar to the target word,
            ordered by their Euclidean distance from `tgt_embedding`.
        """
        sorted_embeddings = sorted(self.__glove_embeddings.keys(),
                                   key=lambda word:
                                   euclidean(self.__glove_embeddings[word],
                                             tgt_embedding))
        return sorted_embeddings[:n_words]
    
    def __create_matrix__(self)->np.ndarray:
        """Starting from the stored vocabulary, it defines and populates the 
        embedding matrix where each entry corresponds to a vocabulary token.

        Returns
        -------
        np.ndarray
            The embedding matrix has shape `(vocabulary length, embedding
            length)`, where the embedding vector length corresponds to the value
            stored in `self.EMBEDDING_DIM`. 
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
                     split_size:float=0.3,
                     rnd_state:bool=True):

        data = {}
        if rnd_state:
            rs = 42
        else:
            rs=None

        # Splitting input data
        (X_train, X_test,
         y_train, y_test) = train_test_split(x, y,
                                             test_size=split_size,
                                             random_state=rs)

        data['x_train'] = X_train
        data['x_test'] = X_test
        data['y_train'] = y_train
        data['y_test'] = y_test
        
        # Data tokenization
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(pd.concat([X_train, X_test]))

        # Vectorizing data to keep `MAX_SEQUENCE_LENGTH` words per sample
        X_train_vect = pad_sequences(tokenizer.texts_to_sequences(X_train),
                                     maxlen = self.MAX_SEQUENCE_LENGTH,
                                     padding = "post",
                                     truncating = "post",
                                     value = 0.)
        X_test_vect = pad_sequences(tokenizer.texts_to_sequences(X_test),
                                    maxlen = self.MAX_SEQUENCE_LENGTH,
                                    padding = "post",
                                    truncating = "post",
                                    value = 0.)
        
        data['embed_x_train'] = X_train_vect
        data['embed_x_test'] = X_test_vect
        self.data = data

        # Vocabulary
        index_word = tokenizer.index_word
        self.vocabulary = index_word
        
        embedding_matrix = self.__create_matrix__()
        self.embedding_matrix = embedding_matrix

################################################################################
# Word2Vec
def build_w2v_model(train_set:pd.DataFrame, feature:str, model_path:str,
                    model_name:str, vector_size:int=200, window:int=2):
    """ It builds, trains and saves a Word2vec model based on the parameter in input

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
    data = train_set[feature] ##attenzione qua

    model_w2v = gensim.models.Word2Vec(
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
    # Saving model
    word2vec_model_file = model_path + model_name + '.model' ##
    model_w2v.save(word2vec_model_file)
    print("The word2vec model has been trained and saved")
    words = list(model_w2v.wv.index_to_key)
    print('Number of words in the vocabulary:')
    print(len(words))

def load_w2v_model(model_path:str):
    model_w2v = gensim.models.Word2Vec.load(model_path)
    return model_w2v

def write_w2v_csv(data:pd.DataFrame, model, path:str, feature:str, vector_size:int=200):
    """ It creates and saves a pd.DataFrame() containing the embedding vector of each tokenixed
        document present in the Dataframe in input.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame in input
    model : _type_
        A gensim.models.Word2vec document already trained
    path : str
        Path where to save the .csv file generated by the function
    feature : str        
        Column of the DataFrame in input to vectorize e.g. 'tokenized_text'
    vector_size : int, optional
        Number of columns of the embedding dataset, by default 200
    """
  # Store the vectors for data in .csv file saved in a specific path
    words = list(model.wv.index_to_key)

    with open(path, 'w+') as word2vec_file:
        for index, row in data.iterrows():
            # Text embedding computed as the mean of the embedding vector of 
            # each word in the document
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