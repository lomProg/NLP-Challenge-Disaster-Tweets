from sklearn.model_selection import train_test_split

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