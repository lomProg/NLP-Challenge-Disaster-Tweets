import re
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

class DataGenerator(object):


    def __init__(self) -> None:
        pass

    def split_data(
            self, x:pd.Series, y:pd.Series,
            reset_index:bool=False, **kwargs) -> None:

        data = {}

        train_sz = None
        if any(re.search("test_size", k) for k in kwargs.keys()):
            test_sz = kwargs.pop("test_size")
        elif any(re.search("train_size", k) for k in kwargs.keys()):
            train_sz = kwargs.pop("train_size")
            test_sz = 1.0 - train_sz
        else:
            test_sz = 0.3
        rnd_st = kwargs.pop("random_state", None)

        # Splitting input data
        (X_train, X_test,
         y_train, y_test) = train_test_split(x, y,
                                             test_size = test_sz,
                                             train_size = train_sz,
                                             random_state = rnd_st,
                                             **kwargs)
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
    
    def tokenize_data(self, *args, max_sequence_length:int=50) -> None:

        if not args and not hasattr(self, "data"):
            raise AttributeError("No text to tokenize")
        elif not hasattr(self, "data"):
            data = {}
        else:
            data = self.data
            args = (data["x_train"], data["x_test"])

        # Data tokenization
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(pd.concat([*args]))

        # Vectorizing data to keep `max_sequence_length` words per sample
        for i in range(len(args)):
            data[f'vect_set_{i}'] = pad_sequences(
                tokenizer.texts_to_sequences(args[i]),
                maxlen = max_sequence_length,
                padding = "post",
                truncating = "post",
                value = 0.)
        self.data = data

def split_train_test(data, test_size = 0.3, shuffle = True):
    X_train, X_test, Y_train, Y_test = train_test_split(data[['cleaned_text',
                                                              'lemmatized_text',
                                                              'stemmed_text',
                                                              'tokenized_text']], 
                                                        data['target'], 
                                                        shuffle = shuffle,
                                                        test_size = test_size, 
                                                        random_state = 15)
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.reset_index()

    return X_train, X_test, Y_train, Y_test