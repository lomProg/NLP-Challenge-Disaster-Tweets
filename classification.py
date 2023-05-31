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
        assert (not hasattr(self, "data") or
                (hasattr(self, "data") and "x_train" not in self.data)), (
            "The data have already been split. The different sets can be "
            "found stored in the data property."
                )

        if hasattr(self, "data"):
            data = {}
            x, y, x_vect = self.data.values()
        elif hasattr(self, "__raw_data__"):
            data = {}
            x, y = self.__raw_data__.values()

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
        if hasattr(self, "data"):
            train_idx = X_train.index
            test_idx = X_test.index
            X_train_vect = [elm for i, elm in enumerate(x_vect)
                            if i in train_idx]
            X_test_vect = [elm for i, elm in enumerate(x_vect)
                            if i in test_idx]
            data['vect_x_train'] = pd.Series(X_train_vect, index=train_idx)
            data['vect_x_test'] = pd.Series(X_test_vect, index=test_idx)
        self.data = data
    
    def tokenize_data(self,
                      max_sequence_length:int=50,
                      **kwargs) -> None:
        assert (not hasattr(self, "data") or
                (hasattr(self, "data") and
                 not any(k.startswith("vect") for k in self.data.keys()))), (
            "The data have already been tokenized. It can be found stored in "
            "the data property.")
        
        if hasattr(self, "data"):
            data = self.data.copy()
            xs = (data["x_train"], data["x_test"])
        elif hasattr(self, "__raw_data__"):
            data = self.__raw_data__.copy()
            xs = (data["x"],)
        data_mapping = dict(zip(range(len(data.keys())), data.keys()))

        # Data tokenization
        # tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(pd.concat([*args]))
        vectorizer = TextVectorization(
            output_sequence_length = max_sequence_length,
            **kwargs)
        vectorizer.adapt(pd.concat([*xs]))

        # Vectorizing data to keep `max_sequence_length` words per sample
        for i, arg in enumerate(xs):
            # seqs = tokenizer.texts_to_sequences(arg)
            # data[f'vect_set_{i}'] = pad_sequences(
            #     seqs,
            #     maxlen = max_sequence_length,
            #     padding = "post",
            #     truncating = "post",
            #     value = 0.)
            set_i = data_mapping[i]
            vect_i = vectorizer(np.array([[s] for s in arg])).numpy()
            data[f'vect_{set_i}'] = pd.Series(vect_i.tolist(),
                                              data[set_i].index)
        self.data = data

        # Vocabulary
        # index_word = tokenizer.index_word
        index_word = dict(zip(range(len(vectorizer.get_vocabulary())),
                              vectorizer.get_vocabulary()))
        self.vocabulary = index_word