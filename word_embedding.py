import numpy as np
from scipy.spatial.distance import euclidean

def load_glove(path:str)->dict:
    """ Loads Glove word embeddings in memory from the file stored in `path`. """
    embeddings_dict={}
    with open(path,'rb') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def find_closest_embeddings(tgt_embedding:np.ndarray,
                            embeddings_dict:dict,
                            n_words:int=5)->list:
    """Generates an ordered list of the `n_words` words in `embeddings_dict`
    most similar to the target word `tgt_embedding` given as input.

    Parameters
    ----------
    tgt_embedding : np.ndarray
        Target word from which to find the most similar words
    embeddings_dict : dict
        Dictionary of all vocabulary embeddings
    n_words : int, optional
        Desired number of similar words to be returned, by default 5

    Returns
    -------
    list
        It collects the `n_words` words most similar to the target word, ordered
        by their Euclidean distance from `tgt_embedding`.
    """
    sorted_embeddings = sorted(embeddings_dict.keys(),
                               key=lambda word: euclidean(embeddings_dict[word],
                                                          tgt_embedding))
    return sorted_embeddings[:n_words]