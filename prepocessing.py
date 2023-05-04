import re
import nltk
import gensim
from typing import List
from nltk.stem import WordNetLemmatizer
from gensim.parsing.porter import PorterStemmer


from text_utils import PUNCTUATIONS, STOPWORDS, EMOTICONS, POS_EMOT, NEG_EMOT, POS_EMOJI, NEG_EMOJI

def extract_hashtag(text:str)->List[str]:
    """It extracts all the hashtags from the text and it collects all of them in a list

    Parameters
    ----------
    text : str
        Text to preprocess

    Returns
    -------
    List[str]
        List of hashtag without the '#'
    """
    hashtag = re.findall(r"#(\w+)", text)
    if hashtag:
        #check if list is not empty
        hashtag = [w.lower() for w in hashtag] #lower case
    return hashtag

def convert_emoticons(text):
    for e in EMOTICONS:
        if EMOTICONS[e] in POS_EMOT:
            text = re.sub(u'(' + e + ')', 'positive', text)
        else:
            text = re.sub(u'(' + e + ')', 'negative', text)
    return text

def convert_emoji(text):
    for e in POS_EMOJI:
        text = re.sub(r'(' + e + ')', 'positive', text)
    for e in NEG_EMOJI:
        text = re.sub(r'(' + e + ')', 'negative', text)
    return text

def remove_neutral_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_whitespaces(text):
    """ The blank spaces are removed from the input text. """
    return " ".join(text.split())

def remove_punctuation(text:str, extra_punc:str = None, wanted_punc:str = None)->str:
    """It removes punctuations marks from input text, adding an additional
    set of symbols if specified.

    Parameters
    ----------
    text : str
        _description_
    extra_punc : str, optional
        _description_, by default None
    wanted_punc : str, optional
        _description_, by default None

    Returns
    -------
    str
        _description_
    """
    punc = PUNCTUATIONS
    if extra_punc:
        punc += extra_punc
    if wanted_punc:
        for c in wanted_punc:
            punc = punc.replace(c, '')
    trans = str.maketrans(dict.fromkeys(punc, ' '))
    return text.translate(trans)

def remove_stopwords(text:str, s_w:set = STOPWORDS)->str:
    """It removes the entire set of stopwords s_w from the input text.

    Parameters
    ----------
    text : str
        _description_
    s_w : set, optional
        _description_, by default STOPWORDS

    Returns
    -------
    str
        _description_
    """
    if isinstance(text, str):
        return ' '.join([word for word in text.split() if word not in s_w])

def clean_text(text:str):
    out_text = text

def remove_digit(text:str)->str:
    """Remove digits from text

    Parameters
    ----------
    text : str
        _description_

    Returns
    -------
    str
        _description_
    """
    return ''.join([i for i in text if not i.isdigit()])

def lemmatization(text:str)->str:
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def stemming(text:str)->str:
    stemmer = PorterStemmer()
    return stemmer.stem_sentence(text)

