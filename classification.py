import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from keras.preprocessing.text import Tokenizer
# from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import TextVectorization

class DataGenerator(object):


    def __init__(self, x:pd.Series, y:pd.Series) -> None:
        data = {"x": x, "y": y}
        self.__raw_data__ = data

    def split_data(self,
                   reset_index:bool=False,
                   **kwargs) -> None:

        x, y = self.__raw_data__.values()
        data = {}

        # Splitting input data
        X_train, X_test, y_train, y_test = train_test_split(x, y, **kwargs)

        if reset_index:
            X_train = X_train.reset_index()
            X_test = X_test.reset_index()
            y_train = y_train.reset_index()
            y_test = y_test.reset_index()

        data['x_train'] = X_train
        data['x_test'] = X_test
        data['y_train'] = y_train
        data['y_test'] = y_test
        self.data = data
    
    def tokenize_data(self,
                    #   *args,
                      max_sequence_length:int=50,
                      **kwargs) -> None:

        data = {}
        if hasattr(self, "data"):
            xs = (self.data["x_train"], self.data["x_test"])
        elif hasattr(self, "__raw_data__"):
            xs = (self.__raw_data__["x"])
        # if not args and not hasattr(self, "data"):
        #     raise AttributeError("No text to tokenize")
        # elif not hasattr(self, "data"):
        #     data = {}
        #     for i, arg in enumerate(args):
        #         data[f'set_{i}'] = arg
        # else:
        #     data = self.data
        #     args = (data["x_train"], data["x_test"])

        # Data tokenization
        # tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(pd.concat([*args]))
        vectorizer = TextVectorization(
            output_sequence_length = max_sequence_length,
            **kwargs)
        vectorizer.adapt(pd.concat([*xs])) #pd.concat([*args])

        # Vectorizing data to keep `max_sequence_length` words per sample
        for i, arg in enumerate(xs): #args
            # seqs = tokenizer.texts_to_sequences(arg)
            # data[f'vect_set_{i}'] = pad_sequences(
            #     seqs,
            #     maxlen = max_sequence_length,
            #     padding = "post",
            #     truncating = "post",
            #     value = 0.)
            data[f'vect_set_{i}'] = vectorizer(
                np.array([[s] for s in arg])).numpy()
        self.data = data

        # Vocabulary
        # index_word = tokenizer.index_word
        index_word = dict(zip(range(len(vectorizer.get_vocabulary())),
                              vectorizer.get_vocabulary()))
        self.vocabulary = index_word