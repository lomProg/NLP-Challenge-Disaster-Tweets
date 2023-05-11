import re
from typing import List
from nltk.stem import WordNetLemmatizer
from gensim.parsing.porter import PorterStemmer
import time

from text_utils import PUNCTUATIONS, STOPWORDS, EMOTICONS, POS_EMOT, NEG_EMOT
from text_utils import SLANG, CONTRACTIONS, STMT_EMOJI, ADDITIONAL_EMOJI

def remove_urls(text:str)->str:
    """ The URLs are removed from the input text. """
    return re.sub(r'http\S+', '', text)

def remove_tags(text:str)->str:
    """ The account tags are removed from the input text. """
    return re.sub("(?<!\S)(@.*?)(?=\s)", "", text)

def convert_emoticons(text:str)->str:
    """Converts all the emoticons present in the input text into the
    corresponding token. Token value can be `:positive:`, `:neutral:` or
    `:negative:`.

    Parameters
    ----------
    text : str
        Input text in which to apply the conversion.

    Returns
    -------
    str
        If emoticons were found in the text, the output will contain the
        corresponding tokens instead of symbols. If instead nothing was found,
        the output string matches the input one.
    """
    for e,m in list(EMOTICONS.items()):
        if re.search(fr"((\b|^|\s){e})", text):
            if any(re.match(p, m, re.I) for p in POS_EMOT):
                tkn = ' :positive:'
            elif any(re.match(n, m, re.I) for n in NEG_EMOT):
                tkn = ' :negative:'
            else:
                tkn = ' :neutral:'
            text = re.sub(fr"((\b|^|\s){e})", tkn, text)
    return text

def convert_emoji(text:str)->str:
    """Converts all the emojis in the input text into the corresponding token,
    according to the emoji dictionaries. Token value can be `:positive:`,
    `:neutral:` or `:negative:`.

    Parameters
    ----------
    text : str
        Input text in which to apply the conversion.

    Returns
    -------
    str
        If emojis were found in the text, the output will contain the
        corresponding tokens instead of symbols. If instead nothing was found,
        the output string matches the input one.
    """
    for k, v in STMT_EMOJI.items():
        if re.search(f"{k}", text, flags=re.U):
            text = re.sub(f"{k}", v["emoji_tkn"], text)

    for k, v in ADDITIONAL_EMOJI.items():
        if re.search(f"{k}", text, flags=re.U):
            text = re.sub(f"{k}", v["emoji_tkn"], text)
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

def convert_slang(text:str)->str:
    """Decontracts colloquial slang in extended form.
    e.g. 'U' is converted in 'You'

    Parameters
    ----------
    text : str
        Input text in which to apply the conversion.

    Returns
    -------
    str
        _description_
    """
    new_text = []
    for w in text.split():
        if w.upper() in list(SLANG.keys()):
            new_text.append(SLANG[w.upper()].lower())
        else:
            new_text.append(w)
    return " ".join(new_text)

def lowercase(text):
    """ Convert input text to lowercase. """
    return text.lower()

def decontract_text(text:str)->str:
  """ Decontract english contracted forms to the extended ones.
  e.g. I'm -> I am
  """
  for c in CONTRACTIONS.keys():
    text = re.sub(c, CONTRACTIONS[c], text)
  return text

def remove_digits(text:str)->str:
    """ Remove digits from text. """
    return ''.join([i for i in text if not i.isdigit()])

def remove_stopwords(text:str, s_w:set = STOPWORDS)->str:
    """It removes the entire set of stopwords s_w from the input text.

    Parameters
    ----------
    text : str
        Input text in which to apply the stopwords removal.
    s_w : set, optional
        Set of the stopwords to remove, by default STOPWORDS

    Returns
    -------
    str
        _description_
    """
    if isinstance(text, str):
        return ' '.join([word for word in text.split() if word not in s_w])

def remove_punctuation(text:str,
                       extra_punc:str = None,
                       wanted_punc:str = None)->str:
    """It removes punctuations marks from input text, adding an additional
    set of symbols if specified.

    Parameters
    ----------
    text : str
        Input text in which to apply the removal.
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

def remove_whitespaces(text):
    """ The blank spaces are removed from the input text. """
    return " ".join(text.split())

def clean_text(text:str, verbose:bool=False)->str:
    # Remove URLs
    t = time.time()
    out_text = remove_urls(text)
    d1 = time.time() - t
    # Remove tweet tags
    t = time.time()
    out_text = remove_tags(out_text)
    d12 = time.time() - t
    # Conversion emoticons
    t = time.time()
    out_text = convert_emoticons(out_text)
    d2 = time.time() - t
    # Conversion emoji
    t = time.time()
    out_text = convert_emoji(out_text)
    d10 = time.time() - t
    t = time.time()
    out_text = remove_neutral_emoji(out_text)
    d11 = time.time() - t
    # Decontraction slangs
    t = time.time()
    out_text = convert_slang(out_text)
    d3 = time.time() - t
    # Text to lowercase
    t = time.time()
    out_text = lowercase(out_text)
    d4 = time.time() - t
    # Decontractions of short english form
    t = time.time()
    out_text = decontract_text(out_text)
    d9 = time.time() - t
    # Remove digits
    t = time.time()
    out_text = remove_digits(out_text)
    d5 = time.time() - t
    # Remove stopwords
    t = time.time()
    out_text = remove_stopwords(out_text)
    d6 = time.time() - t
    # Remove punctuation
    t = time.time()
    out_text = remove_punctuation(out_text)
    d7 = time.time() - t
    # Remove whitespaces
    t = time.time()
    out_text = remove_whitespaces(out_text)
    d8 = time.time() - t

    if verbose:
        print(f"URLs:\t{d1}\n", f"TAGs:\t{d1}\n", f"Emoticons:\t{d2}\n",
              f"Emojis:\t{d10}\n", f"Neutral Emojis:\t{d11}\n",
              f"Slang:\t{d3}\n", f"Lowercase:\t{d4}\n",
              f"Decontractions:\t{d9}", f"Digits:\t{d5}\n",
              f"Stopwords:\t{d6}\n", f"Punctuations:\t{d7}\n",
              f"Whitespaces:\t{d8}\n")

    return out_text

##############################
# 
##############################
def lemmatization(text:str)->str:
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def stemming(text:str)->str:
    stemmer = PorterStemmer()
    return stemmer.stem_sentence(text)

def extract_hashtag(text:str)->List[str]:
    """It extracts all the hashtags from the text and it collects all of them in
    a list.

    Parameters
    ----------
    text : str
        Text to preprocess.

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