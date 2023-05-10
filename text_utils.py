import re
import string
import nltk
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup

import emot
from emot.emo_unicode import EMOTICONS_EMO

nltk.download('stopwords')

PUNCTUATIONS = string.punctuation
STOPWORDS = set(stopwords.words('english'))
EMOTICONS = {
    ":(-|‑)?(\)+|\])":"Happy face or very happy face or smiley",
    ":(-|‑)?(3|>)":"Happy face smiley",
    "(8-|:o)\)":"Happy face smiley",
    ":(-|‑)?\}":"Happy face smiley",
    ":(c|\^)?\)":"Happy face smiley",
    "=(\)|\])":"Happy face smiley",
    "(:|8|X)[-|‑]?D":"Laughing, big grin or laugh with glasses",
    "=(D|3)":"Laughing, big grin or laugh with glasses",
    "B\^D":"Laughing, big grin or laugh with glasses",
    ":(-|‑)?(\(|c|<|\[)":"Frown, sad, angry or pouting",
    ":-\|\|":"Frown, sad, angry or pouting",
    ">:(\(|\[)":"Frown, sad, angry or pouting",
    ":-(\{|@)":"Frown, sad, angry or pouting",
    ":'(-|‑)?\(":"Crying",
    ":'(-|‑)?\)":"Tears of happiness",
    "D‑':":"Horror",
    "D:<":"Disgust",
    "D:":"Sadness",
    "D(8|;|=|X)":"Great dismay",
    ":(-|‑)?(O|o)":"Surprise",
    ":(-|‑)?0":"Shock",
    "8‑0":"Yawn",
    ">:O":"Yawn",
    ":(-|‑)?\*":"Kiss",
    ":X":"Kiss",
    "(;|\*)(-|‑)?\)":"Wink or smirk",
    ";(-|‑)?\]":"Wink or smirk",
    ";\^\)":"Wink or smirk",
    ":‑,":"Wink or smirk",
    ";D":"Wink or smirk",
    "(:|X)(-|‑)?P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":(-|‑)?Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":(-|‑)?b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    "d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    "=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    ">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":(-|‑)?\/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    ":-\[\.\]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    r"(>?:|=)\[\(\\\)\]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    "(>:|=)\/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    ":(L|S)":"Skeptical, annoyed, undecided, uneasy or hesitant",
    "=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    ":(-|‑)?\|":"Straight face",
    ":\$":"Embarrassed or blushing",
    ":(-|‑)?(x|#|&)":"Sealed lips or wearing braces or tongue-tied",
    "O:(-|‑)?\)":"Angel, saint or innocent",
    "0:(-|‑)?(\)|3)":"Angel, saint or innocent",
    "0;\^\)":"Angel, saint or innocent",
    "(>|\}|3):(-|‑)?\)":"Evil or devilish",
    ">;\)":"Evil or devilish",
    "\|;‑\)":"Cool",
    "\|‑O":"Bored",
    ":‑J":"Tongue-in-cheek",
    "#‑\)":"Party all night",
    "%(-|‑)?\)":"Drunk or confused",
    ":(-|‑)?#{3}\.{2}":"Being sick",
    "<:‑\|":"Dump",
    "\(>_<\)>?":"Troubled",
    "\(';'\)":"Baby",
    "\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "\(-_-\)z{3}":"Sleeping",
    "\(\^_-\)":"Wink",
    "\(\(\+_\+\)\)":"Confused",
    "\(\+o\+\)":"Confused",
    "\(o\|o\)":"Ultraman",
    "\^_\^":"Joyful",
    "\(\^_\^\)\/":"Joyful",
    "\(\^(O|o)\^\)／":"Joyful",
    "\(__\)":"Kowtow as a sign of respect, or dogeza for apology",
    "_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",
    "<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",
    "<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",
    "m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",
    "m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",
    "\('_'\)":"Sad or Crying",
    "\(\/_;\)":"Sad or Crying",
    "\(T_T\) \(;_;\)":"Sad or Crying",
    "\(;_;":"Sad or Crying",
    "\((;_:|;O;|:_;|ToT)\)":"Sad or Crying",
    ";(-|‑|n)?;":"Sad or Crying",
    "Q(\.|_)?Q":"Sad or Crying",
    "T\.T":"Sad or Crying",
    "\(-(\.|_)-\)":"Shame",
    "\(一一\)":"Shame",
    "\(；一_一\)":"Shame",
    "\(=_=\)":"Tired",
    "\(=\^·\^=\)":"Cat",
    "\(=\^··\^=\)":"Cat",
    "=_\^=":"Cat",
    "\(\._?\.\)":"Looking down",
    "\^m\^":"Giggling with hand covering mouth",
    "\(・・\?":"Confusion",
    "\(\?_\?\)":"Confusion",
    ">\^_\^<":"Normal Laugh",
    "<\^!\^>":"Normal Laugh",
    "\^\/\^":"Normal Laugh",
    "（\*\^_\^\*）":"Normal Laugh",
    "\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
    "\(\^(\.|J|—|-)?\^\)":"Normal Laugh",
    "\(\^_\^\.?\)":"Normal Laugh",
    "\(\*\^\.\^\*\)":"Normal Laugh",
    "\(\^—\^）":"Normal Laugh",
    "\(#\^\.\^#\)":"Normal Laugh",
    "（\^—\^）":"Waving",
    "\(;_;\)\/~{3}":"Waving",
    "\(\^\.\^\)\/~{3}":"Waving",
    "\(-_-\)\/~{3} \(\$·{2}\)\/~{3}":"Waving",
    "\(T(_|o)T\)\/~{3}":"Waving",
    "\(\*\^0\^\*\)":"Excited",
    "\(\*_\*(\)|;)":"Amazed",
    "\(\+_\+\) \(@_@\)":"Amazed",
    "\(\*\^\^\)v":"Laughing,Cheerful",
    "\(\^_\^\)v":"Laughing,Cheerful",
    "\(\(d\[-_-\]b\)\)":"Headphones,Listening to music",
    '\(-"-\)':"Worried",
    "\(ーー;\)":"Worried",
    "\(\^0_0\^\)":"Eyeglasses",
    "\(＾(ｖ|ｕ)＾\)":"Happy",
    "\(\^\)o\(\^\)":"Happy",
    "\(\^(O|o)\^\)":"Happy",
    "\)\^o\^\(":"Happy",
    ":O o_O":"Surprised",
    "o_0":"Surprised",
    "o\.?O":"Surprised",
    "\(o\.o\)":"Surprised",
    "\(\*￣m￣\)":"Dissatisfied",
    "\(‘A`\)":"Snubbed or Deflated"
}
POS_EMOT = [
    r'.*(happy).*', r'.*(laugh).*', r'.*(happiness).*', r'.*(surprise).*',
    r'.*(kiss).*', r'.*(wink).*', r'.*(tongue)(?!-tied).*', r'.*(angel).*',
    r'.*(cool).*', r'.*(party).*', r'.*(baby).*', r'.*(sleeping).*',
    r'.*(confused).*', r'.*(ultraman).*', r'.*(joyful).*', r'.*(waving).*',
    r'.*(excited).*', r'.*(amazed).*', r'.*(headphones).*', r'.*(eyeglasses).*'
]
NEG_EMOT = [
    r'.*(sad).*', r'.*(crying).*', r'.*(horror).*', r'.*(disgust).*',
    r'.*(dismay).*', r'.*(shock).*', r'.*(yawn).*', r'.*(skeptical).*',
    r'.*(straight face).*', r'.*(embarrassed).*', r'.*(sealed lips).*',
    r'.*(evil).*', r'.*(bored).*', r'.*(sick).*', r'.*(dump).*',
    r'.*(troubled).*', r'.*(shame).*', r'.*(tired).*', r'.*(worried).*',
    r'.*(dissatisfied).*', r'.*(snubbed).*', r'.*(nervous).*', r'.*(confusion).*'
]

POS_EMOJI = [
    '\U0001F601', '\U0001F602', '\U0001F603', '\U0001F604', '\U0001F605',
    '\U0001F606', '\U0001F609', '\U0001F60A', '\U0001F60B', '\U0001F60C',
    '\U0001F60D', '\U0001F60F', '\U0001F618', '\U0001F61A', '\U0001F61C',
    '\U0001F61D', '\U0001F64C', '\U0001F600', '\U0001F607', '\U0001F60E',
    '\U0001F525', '\U0001F44D', '\U0001F4AA', '\U0001F4AF'
    ]
NEG_EMOJI = [
    '\U0001F612', '\U0001F613', '\U0001F614', '\U0001F616', '\U0001F61E',
    '\U0001F620', '\U0001F621', '\U0001F622', '\U0001F623', '\U0001F624',
    '\U0001F625', '\U0001F628', '\U0001F629', '\U0001F62A', '\U0001F62B',
    '\U0001F62D', '\U0001F630', '\U0001F631', '\U0001F632', '\U0001F633',
    '\U0001F635', '\U0001F637', '\U0001F645', '\U0001F610', '\U0001F611',
    '\U0001F615', '\U0001F61F', '\U0001F626', '\U0001F627', '\U0001F4A9', 
    '\U0001F44E'
    ]

SLANG = {
    'AFAIK': 'As Far As I Know',
    'AFK': 'Away From Keyboard',
    'ASAP': 'As Soon As Possible',
    'ATK': 'At The Keyboard',
    'ATM': 'At The Moment',
    'A3': 'Anytime, Anywhere, Anyplace',
    'BAK': 'Back At Keyboard',
    'BBL': 'Be Back Later',
    'BBS': 'Be Back Soon',
    'BFN': 'Bye For Now',
    'B4N': 'Bye For Now',
    'BRB': 'Be Right Back',
    'BRT': 'Be Right There',
    'BTW': 'By The Way',
    'B4': 'Before',
    'CU': 'See You',
    'CUL8R': 'See You Later',
    'CYA': 'See You',
    'FAQ': 'Frequently Asked Questions',
    'FC': 'Fingers Crossed',
    'FWIW': "For What It's Worth",
    'FYI': 'For Your Information',
    'GAL': 'Get A Life',
    'GG': 'Good Game',
    'GN': 'Good Night',
    'GMTA': 'Great Minds Think Alike',
    'GR8': 'Great!',
    'G9': 'Genius',
    'IC': 'I See',
    'ICQ': 'I Seek you',
    'ILU': 'ILU: I Love You',
    'IMHO': 'In My Honest Opinion',
    'IMO': 'In My Opinion',
    'IOW': 'In Other Words',
    'IRL': 'In Real Life',
    'KISS': 'Keep It Simple, Stupid',
    'LDR': 'Long Distance Relationship',
    'LMAO': 'Laugh My Aaa Off',
    'LOL': 'Laughing Out Loud',
    'LTNS': 'Long Time No See',
    'L8R': 'Later',
    'MTE': 'My Thoughts Exactly',
    'M8': 'Mate',
    'NRN': 'No Reply Necessary',
    'OIC': 'Oh I See',
    'PITA': 'Pain In The Ass',
    'PRT': 'Party',
    'PRW': 'Parents Are Watching',
    'ROFL': 'Rolling On The Floor Laughing',
    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',
    'ROTFLMAO': 'Rolling On The Floor Laughing My Ass Off',
    'SK8': 'Skate',
    'STATS': 'Your sex and age',
    'ASL': 'Age Sex Location',
    'THX': 'Thank You',
    'TTFN': 'Ta-Ta For Now!',
    'TTYL': 'Talk To You Later',
    'U': 'You',
    'U2': 'You Too',
    'U4E': 'Yours For Ever',
    'WB': 'Welcome Back',
    'WTF': 'What The Fuck',
    'WTG': 'Way To Go!',
    'WUF': 'Where Are You From?',
    'W8': 'Wait'
    }

EMOJI_URL = "https://kt.ijs.si/data/Emoji_sentiment_ranking/"

def scrape_emojis(url:str=EMOJI_URL)->dict:
    """Through scraping from site [1], site related to the article [2], it
    extracts from the emoji sentiment ranking: the relative unicode code, the
    name associated with the emoji and the sentiment values. Each emoji is also
    assigned a token tag relating to its value based on the formula [1].

    Parameters
    ----------
    url : str, optional
        _description_, by default EMOJI_URL

    Returns
    -------
    dict
        _description_

    Notes
    -----
    The formula [1] is expressed as:

    References
    ----------
    .. [1] https://kt.ijs.si/data/Emoji_sentiment_ranking/
    .. [2] Kralj Novak P., Smailović J., Sluban B., Mozetič I. (2015) "Sentiment
       of Emojis". PLOS ONE 10(12): e0144296.
       https://doi.org/10.1371/journal.pone.0144296
    """
    page = requests.get(url, timeout = 15)
    soup = BeautifulSoup(page.text, 'lxml')

    all_emojis = soup.find('tbody').findAll('tr')

    sentiment_emoji = {}
    for i in range(len(all_emojis)):

        e = all_emojis[i].findAll('td')
        if len(e[2].text) == 4:
            replc_str = r"\\u00"
        elif len(e[2].text) == 5:
            replc_str = r"\\u0"
        elif len(e[2].text) == 7:
            replc_str = r"\\U000"
        else:
            replc_str = r"\\u"

        e_code = re.sub("0x", replc_str, e[2].text)
        e_name = '_'.join(e[-2].text.lower().split())
        e_sneg = float(e[5].text)
        e_spos = float(e[7].text)
        e_sval = float(e[8].text)
        e_tkn = (":positive:" if e_sval > 1 - (2 * e_spos)
                else ":negative:" if e_sval < -1 + (2 * e_sneg)
                else ":neutral:")

        sentiment_emoji[e_code] = {
            "meaning":e_name,
            "sentiment_score":e_sval,
            "emoji_tkn":e_tkn
            }
    return sentiment_emoji

STMT_EMOJI = {
    '\\U0001f602': {
        'meaning': 'face_with_tears_of_joy',
        'sentiment_score': 0.221,
        'emoji_tkn': ':positive:'
        },
    '\\u2764': {
        'meaning': 'heavy_black_heart',
        'sentiment_score': 0.746,
        'emoji_tkn': ':positive:'
        },
    '\\u2665': {
        'meaning': 'black_heart_suit',
        'sentiment_score': 0.657,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f60d': {
        'meaning': 'smiling_face_with_heart-shaped_eyes',
        'sentiment_score': 0.678,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f62d': {
        'meaning': 'loudly_crying_face',
        'sentiment_score': -0.093,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f618': {
        'meaning': 'face_throwing_a_kiss',
        'sentiment_score': 0.701,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f60a': {
        'meaning': 'smiling_face_with_smiling_eyes',
        'sentiment_score': 0.644,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f44c': {
        'meaning': 'ok_hand_sign',
        'sentiment_score': 0.563,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f495': {
        'meaning': 'two_hearts',
        'sentiment_score': 0.632,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f44f': {
        'meaning': 'clapping_hands_sign',
        'sentiment_score': 0.52,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f601': {
        'meaning': 'grinning_face_with_smiling_eyes',
        'sentiment_score': 0.449,
        'emoji_tkn': ':positive:'
        },
    '\\u263a': {
        'meaning': 'white_smiling_face',
        'sentiment_score': 0.657,
        'emoji_tkn': ':positive:'
        },
    '\\u2661': {
        'meaning': 'white_heart_suit',
        'sentiment_score': 0.669,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f44d': {
        'meaning': 'thumbs_up_sign',
        'sentiment_score': 0.521,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f629': {
        'meaning': 'weary_face',
        'sentiment_score': -0.368,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f64f': {
        'meaning': 'person_with_folded_hands',
        'sentiment_score': 0.417,
        'emoji_tkn': ':positive:'
        },
    '\\u270c': {
        'meaning': 'victory_hand',
        'sentiment_score': 0.463,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f60f': {
        'meaning': 'smirking_face',
        'sentiment_score': 0.332,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f609': {
        'meaning': 'winking_face',
        'sentiment_score': 0.463,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f64c': {
        'meaning': 'person_raising_both_hands_in_celebration',
        'sentiment_score': 0.559,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f648': {
        'meaning': 'see-no-evil_monkey',
        'sentiment_score': 0.432,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4aa': {
        'meaning': 'flexed_biceps',
        'sentiment_score': 0.555,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f604': {
        'meaning': 'smiling_face_with_open_mouth_and_smiling_eyes',
        'sentiment_score': 0.421,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f612': {
        'meaning': 'unamused_face',
        'sentiment_score': -0.374,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f483': {
        'meaning': 'dancer',
        'sentiment_score': 0.734,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f496': {
        'meaning': 'sparkling_heart',
        'sentiment_score': 0.712,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f603': {
        'meaning': 'smiling_face_with_open_mouth',
        'sentiment_score': 0.557,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f614': {
        'meaning': 'pensive_face',
        'sentiment_score': -0.146,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f631': {
        'meaning': 'face_screaming_in_fear',
        'sentiment_score': 0.19,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f389': {
        'meaning': 'party_popper',
        'sentiment_score': 0.738,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f61c': {
        'meaning': 'face_with_stuck-out_tongue_and_winking_eye',
        'sentiment_score': 0.455,
        'emoji_tkn': ':positive:'
        },
    '\\u262f': {
        'meaning': 'yin_yang',
        'sentiment_score': 0.001,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f338': {
        'meaning': 'cherry_blossom',
        'sentiment_score': 0.65,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f49c': {
        'meaning': 'purple_heart',
        'sentiment_score': 0.654,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f499': {
        'meaning': 'blue_heart',
        'sentiment_score': 0.73,
        'emoji_tkn': ':positive:'
        },
    '\\u2728': {
        'meaning': 'sparkles',
        'sentiment_score': 0.351,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f633': {
        'meaning': 'flushed_face',
        'sentiment_score': 0.018,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f497': {
        'meaning': 'growing_heart',
        'sentiment_score': 0.657,
        'emoji_tkn': ':positive:'
        },
    '\\u2605': {
        'meaning': 'black_star',
        'sentiment_score': 0.283,
        'emoji_tkn': ':neutral:'
        },
    '\\u2588': {
        'meaning': 'full_block',
        'sentiment_score': -0.032,
        'emoji_tkn': ':neutral:'
        },
    '\\u2600': {
        'meaning': 'black_sun_with_rays',
        'sentiment_score': 0.465,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f621': {
        'meaning': 'pouting_face',
        'sentiment_score': -0.173,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f60e': {
        'meaning': 'smiling_face_with_sunglasses',
        'sentiment_score': 0.491,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f622': {
        'meaning': 'crying_face',
        'sentiment_score': 0.007,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f48b': {
        'meaning': 'kiss_mark',
        'sentiment_score': 0.691,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f60b': {
        'meaning': 'face_savouring_delicious_food',
        'sentiment_score': 0.631,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f64a': {
        'meaning': 'speak-no-evil_monkey',
        'sentiment_score': 0.459,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f634': {
        'meaning': 'sleeping_face',
        'sentiment_score': -0.08,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3b6': {
        'meaning': 'multiple_musical_notes',
        'sentiment_score': 0.537,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f49e': {
        'meaning': 'revolving_hearts',
        'sentiment_score': 0.739,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f60c': {
        'meaning': 'relieved_face',
        'sentiment_score': 0.482,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f525': {
        'meaning': 'fire',
        'sentiment_score': 0.139,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4af': {
        'meaning': 'hundred_points_symbol',
        'sentiment_score': 0.12,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f52b': {
        'meaning': 'pistol',
        'sentiment_score': -0.194,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f49b': {
        'meaning': 'yellow_heart',
        'sentiment_score': 0.709,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f481': {
        'meaning': 'information_desk_person',
        'sentiment_score': 0.326,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f49a': {
        'meaning': 'green_heart',
        'sentiment_score': 0.656,
        'emoji_tkn': ':positive:'
        },
    '\\u266b': {
        'meaning': 'beamed_eighth_notes',
        'sentiment_score': 0.287,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f61e': {
        'meaning': 'disappointed_face',
        'sentiment_score': -0.118,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f606': {
        'meaning': 'smiling_face_with_open_mouth_and_tightly-closed_eyes',
        'sentiment_score': 0.409,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f61d': {
        'meaning': 'face_with_stuck-out_tongue_and_tightly-closed_eyes',
        'sentiment_score': 0.423,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f62a': {
        'meaning': 'sleepy_face',
        'sentiment_score': -0.08,
        'emoji_tkn': ':neutral:'
        },
    '\\ufffd': {
        'meaning': 'replacement_character',
        'sentiment_score': 0.086,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f62b': {
        'meaning': 'tired_face',
        'sentiment_score': -0.145,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f605': {
        'meaning': 'smiling_face_with_open_mouth_and_cold_sweat',
        'sentiment_score': 0.178,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f44a': {
        'meaning': 'fisted_hand_sign',
        'sentiment_score': 0.228,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f480': {
        'meaning': 'skull',
        'sentiment_score': -0.207,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f600': {
        'meaning': 'grinning_face',
        'sentiment_score': 0.568,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f61a': {
        'meaning': 'kissing_face_with_closed_eyes',
        'sentiment_score': 0.71,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f63b': {
        'meaning': 'smiling_cat_face_with_heart-shaped_eyes',
        'sentiment_score': 0.619,
        'emoji_tkn': ':positive:'
        },
    '\\u00a9': {
        'meaning': 'copyright_sign',
        'sentiment_score': 0.117,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f440': {
        'meaning': 'eyes',
        'sentiment_score': 0.063,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f498': {
        'meaning': 'heart_with_arrow',
        'sentiment_score': 0.683,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f413': {
        'meaning': 'rooster',
        'sentiment_score': 0.028,
        'emoji_tkn': ':neutral:'
        },
    '\\u2615': {
        'meaning': 'hot_beverage',
        'sentiment_score': 0.244,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f44b': {
        'meaning': 'waving_hand_sign',
        'sentiment_score': 0.413,
        'emoji_tkn': ':positive:'
        },
    '\\u270b': {
        'meaning': 'raised_hand',
        'sentiment_score': 0.126,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f38a': {
        'meaning': 'confetti_ball',
        'sentiment_score': 0.721,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f355': {
        'meaning': 'slice_of_pizza',
        'sentiment_score': 0.417,
        'emoji_tkn': ':positive:'
        },
    '\\u2744': {
        'meaning': 'snowflake',
        'sentiment_score': 0.506,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f625': {
        'meaning': 'disappointed_but_relieved_face',
        'sentiment_score': 0.122,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f615': {
        'meaning': 'confused_face',
        'sentiment_score': -0.397,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f4a5': {
        'meaning': 'collision_symbol',
        'sentiment_score': 0.148,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f494': {
        'meaning': 'broken_heart',
        'sentiment_score': -0.121,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f624': {
        'meaning': 'face_with_look_of_triumph',
        'sentiment_score': -0.209,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f608': {
        'meaning': 'smiling_face_with_horns',
        'sentiment_score': 0.265,
        'emoji_tkn': ':positive:'
        },
    '\\u25ba': {
        'meaning': 'black_right-pointing_pointer',
        'sentiment_score': 0.162,
        'emoji_tkn': ':neutral:'
        },
    '\\u2708': {
        'meaning': 'airplane',
        'sentiment_score': 0.415,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f51d': {
        'meaning': 'top_with_upwards_arrow_above',
        'sentiment_score': 0.474,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f630': {
        'meaning': 'face_with_open_mouth_and_cold_sweat',
        'sentiment_score': -0.02,
        'emoji_tkn': ':neutral:'
        },
    '\\u26bd': {
        'meaning': 'soccer_ball',
        'sentiment_score': 0.616,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f611': {
        'meaning': 'expressionless_face',
        'sentiment_score': -0.311,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f451': {
        'meaning': 'crown',
        'sentiment_score': 0.694,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f639': {
        'meaning': 'cat_face_with_tears_of_joy',
        'sentiment_score': 0.141,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f449': {
        'meaning': 'white_right_pointing_backhand_index',
        'sentiment_score': 0.39,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f343': {
        'meaning': 'leaf_fluttering_in_wind',
        'sentiment_score': 0.378,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f381': {
        'meaning': 'wrapped_present',
        'sentiment_score': 0.759,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f620': {
        'meaning': 'angry_face',
        'sentiment_score': -0.299,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f427': {
        'meaning': 'penguin',
        'sentiment_score': 0.456,
        'emoji_tkn': ':positive:'
        },
    '\\u2606': {
        'meaning': 'white_star',
        'sentiment_score': 0.428,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f340': {
        'meaning': 'four_leaf_clover',
        'sentiment_score': 0.285,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f388': {
        'meaning': 'balloon',
        'sentiment_score': 0.718,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f385': {
        'meaning': 'father_christmas',
        'sentiment_score': 0.318,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f613': {
        'meaning': 'face_with_cold_sweat',
        'sentiment_score': -0.08,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f623': {
        'meaning': 'persevering_face',
        'sentiment_score': -0.212,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f610': {
        'meaning': 'neutral_face',
        'sentiment_score': -0.388,
        'emoji_tkn': ':negative:'
        },
    '\\u270a': {
        'meaning': 'raised_fist',
        'sentiment_score': 0.429,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f628': {
        'meaning': 'fearful_face',
        'sentiment_score': -0.14,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f616': {
        'meaning': 'confounded_face',
        'sentiment_score': -0.155,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f4a4': {
        'meaning': 'sleeping_symbol',
        'sentiment_score': 0.37,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f493': {
        'meaning': 'beating_heart',
        'sentiment_score': 0.664,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f44e': {
        'meaning': 'thumbs_down_sign',
        'sentiment_score': -0.188,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f4a6': {
        'meaning': 'splashing_sweat_symbol',
        'sentiment_score': 0.471,
        'emoji_tkn': ':positive:'
        },
    '\\u2714': {
        'meaning': 'heavy_check_mark',
        'sentiment_score': 0.27,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f637': {
        'meaning': 'face_with_medical_mask',
        'sentiment_score': -0.169,
        'emoji_tkn': ':negative:'
        },
    '\\u26a1': {
        'meaning': 'high_voltage_sign',
        'sentiment_score': 0.177,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f64b': {
        'meaning': 'happy_person_raising_one_hand',
        'sentiment_score': 0.485,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f384': {
        'meaning': 'christmas_tree',
        'sentiment_score': 0.531,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4a9': {
        'meaning': 'pile_of_poo',
        'sentiment_score': -0.116,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3b5': {
        'meaning': 'musical_note',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\u27a1': {
        'meaning': 'black_rightwards_arrow',
        'sentiment_score': 0.147,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f61b': {
        'meaning': 'face_with_stuck-out_tongue',
        'sentiment_score': 0.601,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f62c': {
        'meaning': 'grimacing_face',
        'sentiment_score': 0.194,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f46f': {
        'meaning': 'woman_with_bunny_ears',
        'sentiment_score': 0.439,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f48e': {
        'meaning': 'gem_stone',
        'sentiment_score': 0.561,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f33f': {
        'meaning': 'herb',
        'sentiment_score': 0.384,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f382': {
        'meaning': 'birthday_cake',
        'sentiment_score': 0.613,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f31f': {
        'meaning': 'glowing_star',
        'sentiment_score': 0.327,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f52e': {
        'meaning': 'crystal_ball',
        'sentiment_score': 0.267,
        'emoji_tkn': ':neutral:'
        },
    '\\u2757': {
        'meaning': 'heavy_exclamation_mark_symbol',
        'sentiment_score': 0.1,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f46b': {
        'meaning': 'man_and_woman_holding_hands',
        'sentiment_score': 0.255,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3c6': {
        'meaning': 'trophy',
        'sentiment_score': 0.726,
        'emoji_tkn': ':positive:'
        },
    '\\u2716': {
        'meaning': 'heavy_multiplication_x',
        'sentiment_score': 0.311,
        'emoji_tkn': ':positive:'
        },
    '\\u261d': {
        'meaning': 'white_up_pointing_index',
        'sentiment_score': 0.309,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f619': {
        'meaning': 'kissing_face_with_smiling_eyes',
        'sentiment_score': 0.778,
        'emoji_tkn': ':positive:'
        },
    '\\u26c4': {
        'meaning': 'snowman_without_snow',
        'sentiment_score': 0.521,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f445': {
        'meaning': 'tongue',
        'sentiment_score': 0.461,
        'emoji_tkn': ':positive:'
        },
    '\\u266a': {
        'meaning': 'eighth_note',
        'sentiment_score': 0.534,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f342': {
        'meaning': 'fallen_leaf',
        'sentiment_score': 0.547,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f48f': {
        'meaning': 'kiss',
        'sentiment_score': 0.388,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f52a': {
        'meaning': 'hocho',
        'sentiment_score': 0.07,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f334': {
        'meaning': 'palm_tree',
        'sentiment_score': 0.525,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f448': {
        'meaning': 'white_left_pointing_backhand_index',
        'sentiment_score': 0.424,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f339': {
        'meaning': 'rose',
        'sentiment_score': 0.6,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f646': {
        'meaning': 'face_with_ok_gesture',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\u279c': {
        'meaning': 'heavy_round-tipped_rightwards_arrow',
        'sentiment_score': 0.162,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f47b': {
        'meaning': 'ghost',
        'sentiment_score': 0.228,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4b0': {
        'meaning': 'money_bag',
        'sentiment_score': 0.251,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f37b': {
        'meaning': 'clinking_beer_mugs',
        'sentiment_score': 0.512,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f645': {
        'meaning': 'face_with_no_good_gesture',
        'sentiment_score': -0.202,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f31e': {
        'meaning': 'sun_with_face',
        'sentiment_score': 0.558,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f341': {
        'meaning': 'maple_leaf',
        'sentiment_score': 0.482,
        'emoji_tkn': ':positive:'
        },
    '\\u2b50': {
        'meaning': 'white_medium_star',
        'sentiment_score': 0.58,
        'emoji_tkn': ':positive:'
        },
    '\\u25aa': {
        'meaning': 'black_small_square',
        'sentiment_score': 0.198,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f380': {
        'meaning': 'ribbon',
        'sentiment_score': 0.629,
        'emoji_tkn': ':positive:'
        },
    '\\u2501': {
        'meaning': 'box_drawings_heavy_horizontal',
        'sentiment_score': 0.176,
        'emoji_tkn': ':neutral:'
        },
    '\\u2637': {
        'meaning': 'trigram_for_earth',
        'sentiment_score': 0.064,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f437': {
        'meaning': 'pig_face',
        'sentiment_score': 0.368,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f649': {
        'meaning': 'hear-no-evil_monkey',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f33a': {
        'meaning': 'hibiscus',
        'sentiment_score': 0.549,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f485': {
        'meaning': 'nail_polish',
        'sentiment_score': 0.388,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f436': {
        'meaning': 'dog_face',
        'sentiment_score': 0.576,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f31a': {
        'meaning': 'new_moon_with_face',
        'sentiment_score': 0.464,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f47d': {
        'meaning': 'extraterrestrial_alien',
        'sentiment_score': 0.315,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3a4': {
        'meaning': 'microphone',
        'sentiment_score': 0.476,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f46d': {
        'meaning': 'two_women_holding_hands',
        'sentiment_score': 0.463,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3a7': {
        'meaning': 'headphone',
        'sentiment_score': 0.414,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f446': {
        'meaning': 'white_up_pointing_backhand_index',
        'sentiment_score': 0.326,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f378': {
        'meaning': 'cocktail_glass',
        'sentiment_score': 0.539,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f377': {
        'meaning': 'wine_glass',
        'sentiment_score': 0.393,
        'emoji_tkn': ':positive:'
        },
    '\\u00ae': {
        'meaning': 'registered_sign',
        'sentiment_score': 0.279,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f349': {
        'meaning': 'watermelon',
        'sentiment_score': 0.597,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f607': {
        'meaning': 'smiling_face_with_halo',
        'sentiment_score': 0.587,
        'emoji_tkn': ':positive:'
        },
    '\\u2611': {
        'meaning': 'ballot_box_with_check',
        'sentiment_score': 0.101,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3c3': {
        'meaning': 'runner',
        'sentiment_score': 0.406,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f63f': {
        'meaning': 'crying_cat_face',
        'sentiment_score': -0.372,
        'emoji_tkn': ':negative:'
        },
    '\\u2502': {
        'meaning': 'box_drawings_light_vertical',
        'sentiment_score': 0.343,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4a3': {
        'meaning': 'bomb',
        'sentiment_score': 0.007,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f37a': {
        'meaning': 'beer_mug',
        'sentiment_score': 0.493,
        'emoji_tkn': ':positive:'
        },
    '\\u25b6': {
        'meaning': 'black_right-pointing_triangle',
        'sentiment_score': 0.209,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f632': {
        'meaning': 'astonished_face',
        'sentiment_score': -0.068,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3b8': {
        'meaning': 'guitar',
        'sentiment_score': 0.516,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f379': {
        'meaning': 'tropical_drink',
        'sentiment_score': 0.659,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4ab': {
        'meaning': 'dizzy_symbol',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4da': {
        'meaning': 'books',
        'sentiment_score': 0.336,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f636': {
        'meaning': 'face_without_mouth',
        'sentiment_score': -0.142,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f337': {
        'meaning': 'tulip',
        'sentiment_score': 0.538,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f49d': {
        'meaning': 'heart_with_ribbon',
        'sentiment_score': 0.644,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4a8': {
        'meaning': 'dash_symbol',
        'sentiment_score': 0.381,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3c8': {
        'meaning': 'american_football',
        'sentiment_score': 0.53,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f48d': {
        'meaning': 'ring',
        'sentiment_score': 0.478,
        'emoji_tkn': ':positive:'
        },
    '\\u2614': {
        'meaning': 'umbrella_with_rain_drops',
        'sentiment_score': 0.289,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f478': {
        'meaning': 'princess',
        'sentiment_score': 0.605,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f1ea': {
        'meaning': 'regional_indicator_symbol_letter_e',
        'sentiment_score': 0.616,
        'emoji_tkn': ':positive:'
        },
    '\\u2591': {
        'meaning': 'light_shade',
        'sentiment_score': -0.045,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f369': {
        'meaning': 'doughnut',
        'sentiment_score': 0.382,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f47e': {
        'meaning': 'alien_monster',
        'sentiment_score': 0.361,
        'emoji_tkn': ':positive:'
        },
    '\\u2601': {
        'meaning': 'cloud',
        'sentiment_score': 0.308,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f33b': {
        'meaning': 'sunflower',
        'sentiment_score': 0.57,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f635': {
        'meaning': 'dizzy_face',
        'sentiment_score': 0.085,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4d2': {
        'meaning': 'ledger',
        'sentiment_score': 0.038,
        'emoji_tkn': ':neutral:'
        },
    '\\u21bf': {
        'meaning': 'upwards_harpoon_with_barb_leftwards',
        'sentiment_score': 0.648,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f42f': {
        'meaning': 'tiger_face',
        'sentiment_score': 0.476,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f47c': {
        'meaning': 'baby_angel',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f354': {
        'meaning': 'hamburger',
        'sentiment_score': 0.277,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f638': {
        'meaning': 'grinning_cat_face_with_smiling_eyes',
        'sentiment_score': 0.41,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f476': {
        'meaning': 'baby',
        'sentiment_score': 0.434,
        'emoji_tkn': ':positive:'
        },
    '\\u21be': {
        'meaning': 'upwards_harpoon_with_barb_rightwards',
        'sentiment_score': 0.646,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f490': {
        'meaning': 'bouquet',
        'sentiment_score': 0.735,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f30a': {
        'meaning': 'water_wave',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f366': {
        'meaning': 'soft_ice_cream',
        'sentiment_score': 0.459,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f353': {
        'meaning': 'strawberry',
        'sentiment_score': 0.67,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f447': {
        'meaning': 'white_down_pointing_backhand_index',
        'sentiment_score': 0.247,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f486': {
        'meaning': 'face_massage',
        'sentiment_score': 0.221,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f374': {
        'meaning': 'fork_and_knife',
        'sentiment_score': 0.537,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f627': {
        'meaning': 'anguished_face',
        'sentiment_score': -0.063,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f1f8': {
        'meaning': 'regional_indicator_symbol_letter_s',
        'sentiment_score': 0.511,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f62e': {
        'meaning': 'face_with_open_mouth',
        'sentiment_score': 0.269,
        'emoji_tkn': ':positive:'
        },
    '\\u2593': {
        'meaning': 'dark_shade',
        'sentiment_score': 0.011,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f6ab': {
        'meaning': 'no_entry_sign',
        'sentiment_score': -0.44,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f63d': {
        'meaning': 'kissing_cat_face_with_closed_eyes',
        'sentiment_score': 0.571,
        'emoji_tkn': ':positive:'},
    '\\U0001f308': {
        'meaning': 'rainbow',
        'sentiment_score': 0.516,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f640': {
        'meaning': 'weary_cat_face',
        'sentiment_score': 0.33,
        'emoji_tkn': ':positive:'
        },
    '\\u26a0': {
        'meaning': 'warning_sign',
        'sentiment_score': -0.066,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3ae': {
        'meaning': 'video_game',
        'sentiment_score': 0.427,
        'emoji_tkn': ':positive:'
        },
    '\\u256f': {
        'meaning': 'box_drawings_light_arc_up_and_left',
        'sentiment_score': -0.011,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f346': {
        'meaning': 'aubergine',
        'sentiment_score': 0.402,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f370': {
        'meaning': 'shortcake',
        'sentiment_score': 0.448,
        'emoji_tkn': ':positive:'
        },
    '\\u2713': {
        'meaning': 'check_mark',
        'sentiment_score': 0.287,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f450': {
        'meaning': 'open_hands_sign',
        'sentiment_score': -0.023,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f647': {
        'meaning': 'person_bowing_deeply',
        'sentiment_score': 0.14,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f35f': {
        'meaning': 'french_fries',
        'sentiment_score': 0.302,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f34c': {
        'meaning': 'banana',
        'sentiment_score': 0.435,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f491': {
        'meaning': 'couple_with_heart',
        'sentiment_score': 0.659,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f46c': {
        'meaning': 'two_men_holding_hands',
        'sentiment_score': -0.059,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f423': {
        'meaning': 'hatching_chick',
        'sentiment_score': 0.476,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f383': {
        'meaning': 'jack-o-lantern',
        'sentiment_score': 0.595,
        'emoji_tkn': ':positive:'
        },
    '\\u25ac': {
        'meaning': 'black_rectangle',
        'sentiment_score': 0.464,
        'emoji_tkn': ':positive:'
        },
    '\\ufffc': {
        'meaning': 'object_replacement_character',
        'sentiment_score': -0.476,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f61f': {
        'meaning': 'worried_face',
        'sentiment_score': 0.072,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f43e': {
        'meaning': 'paw_prints',
        'sentiment_score': 0.605,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f393': {
        'meaning': 'graduation_cap',
        'sentiment_score': 0.563,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3ca': {
        'meaning': 'swimmer',
        'sentiment_score': 0.575,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f36b': {
        'meaning': 'chocolate_bar',
        'sentiment_score': 0.152,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4f7': {
        'meaning': 'camera',
        'sentiment_score': 0.43,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f444': {
        'meaning': 'mouth',
        'sentiment_score': 0.474,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f33c': {
        'meaning': 'blossom',
        'sentiment_score': 0.779,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f6b6': {
        'meaning': 'pedestrian',
        'sentiment_score': -0.143,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f431': {
        'meaning': 'cat_face',
        'sentiment_score': 0.513,
        'emoji_tkn': ':positive:'
        },
    '\\u2551': {
        'meaning': 'box_drawings_double_vertical',
        'sentiment_score': 0.145,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f438': {
        'meaning': 'frog_face',
        'sentiment_score': -0.08,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f1fa': {
        'meaning': 'regional_indicator_symbol_letter_u',
        'sentiment_score': 0.541,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f47f': {
        'meaning': 'imp',
        'sentiment_score': -0.534,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f6ac': {
        'meaning': 'smoking_symbol',
        'sentiment_score': 0.521,
        'emoji_tkn': ':positive:'
        },
    '\\u273f': {
        'meaning': 'black_florette',
        'sentiment_score': 0.384,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4d6': {
        'meaning': 'open_book',
        'sentiment_score': 0.169,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f412': {
        'meaning': 'monkey',
        'sentiment_score': 0.521,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f30d': {
        'meaning': 'earth_globe_europe-africa',
        'sentiment_score': 0.592,
        'emoji_tkn': ':positive:'
        },
    '\\u250a': {
        'meaning': 'box_drawings_light_quadruple_dash_vertical',
        'sentiment_score': 0.958,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f425': {
        'meaning': 'front-facing_baby_chick',
        'sentiment_score': 0.586,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f300': {
        'meaning': 'cyclone',
        'sentiment_score': 0.101,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f43c': {
        'meaning': 'panda_face',
        'sentiment_score': 0.261,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3a5': {
        'meaning': 'movie_camera',
        'sentiment_score': 0.29,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f484': {
        'meaning': 'lipstick',
        'sentiment_score': 0.435,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4b8': {
        'meaning': 'money_with_wings',
        'sentiment_score': 0.159,
        'emoji_tkn': ':neutral:'
        },
    '\\u26d4': {
        'meaning': 'no_entry',
        'sentiment_score': 0.485,
        'emoji_tkn': ':positive:'
        },
    '\\u25cf': {
        'meaning': 'black_circle',
        'sentiment_score': 0.176,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3c0': {
        'meaning': 'basketball_and_hoop',
        'sentiment_score': 0.254,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f489': {
        'meaning': 'syringe',
        'sentiment_score': 0.358,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f49f': {
        'meaning': 'heart_decoration',
        'sentiment_score': 0.682,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f697': {
        'meaning': 'automobile',
        'sentiment_score': 0.231,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f62f': {
        'meaning': 'hushed_face',
        'sentiment_score': 0.123,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4dd': {
        'meaning': 'memo',
        'sentiment_score': 0.231,
        'emoji_tkn': ':neutral:'
        },
    '\\u2550': {
        'meaning': 'box_drawings_double_horizontal',
        'sentiment_score': 0.015,
        'emoji_tkn': ':neutral:'
        },
    '\\u2666': {
        'meaning': 'black_diamond_suit',
        'sentiment_score': 0.453,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4ad': {
        'meaning': 'thought_balloon',
        'sentiment_score': 0.206,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f319': {
        'meaning': 'crescent_moon',
        'sentiment_score': 0.59,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f41f': {
        'meaning': 'fish',
        'sentiment_score': 0.689,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f463': {
        'meaning': 'footprints',
        'sentiment_score': 0.344,
        'emoji_tkn': ':positive:'
        },
    '\\u261e': {
        'meaning': 'white_right_pointing_index',
        'sentiment_score': 0.115,
        'emoji_tkn': ':neutral:'
        },
    '\\u2702': {
        'meaning': 'black_scissors',
        'sentiment_score': -0.459,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f5ff': {
        'meaning': 'moyai',
        'sentiment_score': 0.443,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f35d': {
        'meaning': 'spaghetti',
        'sentiment_score': 0.117,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f46a': {
        'meaning': 'family',
        'sentiment_score': -0.017,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f36d': {
        'meaning': 'lollipop',
        'sentiment_score': 0.3,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f303': {
        'meaning': 'night_with_stars',
        'sentiment_score': 0.39,
        'emoji_tkn': ':positive:'
        },
    '\\u274c': {
        'meaning': 'cross_mark',
        'sentiment_score': 0.271,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f430': {
        'meaning': 'rabbit_face',
        'sentiment_score': 0.586,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f48a': {
        'meaning': 'pill',
        'sentiment_score': 0.431,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f6a8': {
        'meaning': 'police_cars_revolving_light',
        'sentiment_score': 0.638,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f626': {
        'meaning': 'frowning_face_with_open_mouth',
        'sentiment_score': -0.368,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f36a': {
        'meaning': 'cookie',
        'sentiment_score': 0.316,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f363': {
        'meaning': 'sushi',
        'sentiment_score': -0.232,
        'emoji_tkn': ':negative:'
        },
    '\\u256d': {
        'meaning': 'box_drawings_light_arc_down_and_right',
        'sentiment_score': 0.161,
        'emoji_tkn': ':neutral:'
        },
    '\\u2727': {
        'meaning': 'white_four_pointed_star',
        'sentiment_score': 0.321,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f386': {
        'meaning': 'fireworks',
        'sentiment_score': 0.709,
        'emoji_tkn': ':positive:'
        },
    '\\u256e': {
        'meaning': 'box_drawings_light_arc_down_and_left',
        'sentiment_score': 0.127,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f38e': {
        'meaning': 'japanese_dolls',
        'sentiment_score': 0.907,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f1e9': {
        'meaning': 'regional_indicator_symbol_letter_d',
        'sentiment_score': 0.611,
        'emoji_tkn': ':positive:'
        },
    '\\u2705': {
        'meaning': 'white_heavy_check_mark',
        'sentiment_score': 0.407,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f479': {
        'meaning': 'japanese_ogre',
        'sentiment_score': 0.058,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4f1': {
        'meaning': 'mobile_phone',
        'sentiment_score': 0.308,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f64d': {
        'meaning': 'person_frowning',
        'sentiment_score': -0.327,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f351': {
        'meaning': 'peach',
        'sentiment_score': 0.25,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3bc': {
        'meaning': 'musical_score',
        'sentiment_score': 0.327,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f50a': {
        'meaning': 'speaker_with_three_sound_waves',
        'sentiment_score': 0.404,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f30c': {
        'meaning': 'milky_way',
        'sentiment_score': 0.52,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f34e': {
        'meaning': 'red_apple',
        'sentiment_score': 0.32,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f43b': {
        'meaning': 'bear_face',
        'sentiment_score': 0.44,
        'emoji_tkn': ':positive:'
        },
    '\\u2500': {
        'meaning': 'box_drawings_light_horizontal',
        'sentiment_score': 0.14,
        'emoji_tkn': ':neutral:'
        },
    '\\u2570': {
        'meaning': 'box_drawings_light_arc_up_and_right',
        'sentiment_score': -0.06,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f487': {
        'meaning': 'haircut',
        'sentiment_score': 0.327,
        'emoji_tkn': ':positive:'
        },
    '\\u266c': {
        'meaning': 'beamed_sixteenth_notes',
        'sentiment_score': 0.245,
        'emoji_tkn': ':neutral:'
        },
    '\\u265a': {
        'meaning': 'black_chess_king',
        'sentiment_score': 0.041,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f534': {
        'meaning': 'large_red_circle',
        'sentiment_score': 0.396,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f371': {
        'meaning': 'bento_box',
        'sentiment_score': -0.313,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f34a': {
        'meaning': 'tangerine',
        'sentiment_score': 0.417,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f352': {
        'meaning': 'cherries',
        'sentiment_score': 0.313,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f42d': {
        'meaning': 'mouse_face',
        'sentiment_score': 0.688,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f45f': {
        'meaning': 'athletic_shoe',
        'sentiment_score': 0.417,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f30e': {
        'meaning': 'earth_globe_americas',
        'sentiment_score': 0.319,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f34d': {
        'meaning': 'pineapple',
        'sentiment_score': 0.468,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f42e': {
        'meaning': 'cow_face',
        'sentiment_score': 0.587,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4f2': {
        'meaning': 'mobile_phone_with_rightwards_arrow_at_left',
        'sentiment_score': 0.239,
        'emoji_tkn': ':positive:'
        },
    '\\u263c': {
        'meaning': 'white_sun_with_rays',
        'sentiment_score': 0.196,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f305': {
        'meaning': 'sunrise',
        'sentiment_score': 0.356,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f1f7': {
        'meaning': 'regional_indicator_symbol_letter_r',
        'sentiment_score': 0.667,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f460': {
        'meaning': 'high-heeled_shoe',
        'sentiment_score': 0.356,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f33d': {
        'meaning': 'ear_of_maize',
        'sentiment_score': 0.444,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4a7': {
        'meaning': 'droplet',
        'sentiment_score': -0.159,
        'emoji_tkn': ':negative:'
        },
    '\\u2753': {
        'meaning': 'black_question_mark_ornament',
        'sentiment_score': 0.068,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f36c': {
        'meaning': 'candy',
        'sentiment_score': 0.364,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f63a': {
        'meaning': 'smiling_cat_face_with_open_mouth',
        'sentiment_score': 0.395,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f434': {
        'meaning': 'horse_face',
        'sentiment_score': 0.07,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f680': {
        'meaning': 'rocket',
        'sentiment_score': 0.488,
        'emoji_tkn': ':positive:'
        },
    '\\u00a6': {
        'meaning': 'broken_bar',
        'sentiment_score': 0.581,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4a2': {
        'meaning': 'anger_symbol',
        'sentiment_score': 0.233,
        'emoji_tkn': ':positive:'},
    '\\U0001f3ac': {
        'meaning': 'clapper_board',
        'sentiment_score': 0.279,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f367': {
        'meaning': 'shaved_ice',
        'sentiment_score': 0.302,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f35c': {
        'meaning': 'steaming_bowl',
        'sentiment_score': 0.395,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f40f': {
        'meaning': 'ram',
        'sentiment_score': 0.558,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f418': {
        'meaning': 'elephant',
        'sentiment_score': 0.023,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f467': {
        'meaning': 'girl',
        'sentiment_score': 0.14,
        'emoji_tkn': ':neutral:'
        },
    '\\u2800': {
        'meaning': 'braille_pattern_blank',
        'sentiment_score': 0.047,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3c4': {
        'meaning': 'surfer',
        'sentiment_score': 0.524,
        'emoji_tkn': ':positive:'
        },
    '\\u27a4': {
        'meaning': 'black_rightwards_arrowhead',
        'sentiment_score': 0.31,
        'emoji_tkn': ':neutral:'
        },
    '\\u2b06': {
        'meaning': 'upwards_black_arrow',
        'sentiment_score': 0.286,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f34b': {
        'meaning': 'lemon',
        'sentiment_score': 0.244,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f197': {
        'meaning': 'squared_ok',
        'sentiment_score': 0.537,
        'emoji_tkn': ':positive:'
        },
    '\\u26aa': {
        'meaning': 'medium_white_circle',
        'sentiment_score': 0.45,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4fa': {
        'meaning': 'television',
        'sentiment_score': 0.375,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f345': {
        'meaning': 'tomato',
        'sentiment_score': 0.35,
        'emoji_tkn': ':positive:'
        },
    '\\u26c5': {
        'meaning': 'sun_behind_cloud',
        'sentiment_score': 0.45,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f422': {
        'meaning': 'turtle',
        'sentiment_score': 0.2,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f459': {
        'meaning': 'bikini',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3e1': {
        'meaning': 'house_with_garden',
        'sentiment_score': 0.436,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f33e': {
        'meaning': 'ear_of_rice',
        'sentiment_score': 0.538,
        'emoji_tkn': ':positive:'
        },
    '\\u25c9': {
        'meaning': 'fisheye',
        'sentiment_score': 0.256,
        'emoji_tkn': ':neutral:'
        },
    '\\u270f': {
        'meaning': 'pencil',
        'sentiment_score': 0.342,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f42c': {
        'meaning': 'dolphin',
        'sentiment_score': 0.421,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f364': {
        'meaning': 'fried_shrimp',
        'sentiment_score': 0.053,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f1f9': {
        'meaning': 'regional_indicator_symbol_letter_t',
        'sentiment_score': 0.579,
        'emoji_tkn': ':positive:'
        },
    '\\u2663': {
        'meaning': 'black_club_suit',
        'sentiment_score': 0.342,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f41d': {
        'meaning': 'honeybee',
        'sentiment_score': 0.211,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f31d': {
        'meaning': 'full_moon_with_face',
        'sentiment_score': 0.189,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f1ee': {
        'meaning': 'regional_indicator_symbol_letter_i',
        'sentiment_score': 0.595,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f50b': {
        'meaning': 'battery',
        'sentiment_score': -0.486,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f40d': {
        'meaning': 'snake',
        'sentiment_score': 0.351,
        'emoji_tkn': ':positive:'
        },
    '\\u2654': {
        'meaning': 'white_chess_king',
        'sentiment_score': 0.541,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f373': {
        'meaning': 'cooking',
        'sentiment_score': 0.028,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f535': {
        'meaning': 'large_blue_circle',
        'sentiment_score': 0.306,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f63e': {
        'meaning': 'pouting_cat_face',
        'sentiment_score': -0.333,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f315': {
        'meaning': 'full_moon_symbol',
        'sentiment_score': 0.556,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f428': {
        'meaning': 'koala',
        'sentiment_score': 0.444,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f510': {
        'meaning': 'closed_lock_with_key',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4bf': {
        'meaning': 'optical_disc',
        'sentiment_score': 0.611,
        'emoji_tkn': ':positive:'
        },
    '\\u2741': {
        'meaning': 'eight_petalled_outlined_black_florette',
        'sentiment_score': 0.056,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f333': {
        'meaning': 'deciduous_tree',
        'sentiment_score': 0.486,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f470': {
        'meaning': 'bride_with_veil',
        'sentiment_score': 0.486,
        'emoji_tkn': ':positive:'
        },
    '\\u2740': {
        'meaning': 'white_florette',
        'sentiment_score': 0.4,
        'emoji_tkn': ':positive:'
        },
    '\\u2693': {
        'meaning': 'anchor',
        'sentiment_score': 0.571,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f6b4': {
        'meaning': 'bicyclist',
        'sentiment_score': 0.657,
        'emoji_tkn': ':positive:'
        },
    '\\u2580': {
        'meaning': 'upper_half_block',
        'sentiment_score': -0.143,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f457': {
        'meaning': 'dress',
        'sentiment_score': 0.235,
        'emoji_tkn': ':positive:'
        },
    '\\u2795': {
        'meaning': 'heavy_plus_sign',
        'sentiment_score': 0.529,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4ac': {
        'meaning': 'speech_balloon',
        'sentiment_score': 0.364,
        'emoji_tkn': ':positive:'
        },
    '\\u2592': {
        'meaning': 'medium_shade',
        'sentiment_score': -0.03,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f51c': {
        'meaning': 'soon_with_rightwards_arrow_above',
        'sentiment_score': 0.273,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f368': {
        'meaning': 'ice_cream',
        'sentiment_score': 0.212,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4b2': {
        'meaning': 'heavy_dollar_sign',
        'sentiment_score': 0.242,
        'emoji_tkn': ':neutral:'
        },
    '\\u26fd': {
        'meaning': 'fuel_pump',
        'sentiment_score': 0.152,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f359': {
        'meaning': 'rice_ball',
        'sentiment_score': 0.281,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f357': {
        'meaning': 'poultry_leg',
        'sentiment_score': 0.063,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f372': {
        'meaning': 'pot_of_food',
        'sentiment_score': 0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f365': {
        'meaning': 'fish_cake_with_swirl_design',
        'sentiment_score': -0.594,
        'emoji_tkn': ':negative:'
        },
    '\\u25b8': {
        'meaning': 'black_right-pointing_small_triangle',
        'sentiment_score': 0.219,
        'emoji_tkn': ':neutral:'
        },
    '\\u265b': {
        'meaning': 'black_chess_queen',
        'sentiment_score': 0.188,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f63c': {
        'meaning': 'cat_face_with_wry_smile',
        'sentiment_score': 0.355,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f419': {
        'meaning': 'octopus',
        'sentiment_score': 0.387,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f468': {
        'meaning': 'man',
        'sentiment_score': 0.516,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f35a': {
        'meaning': 'cooked_rice',
        'sentiment_score': 0.452,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f356': {
        'meaning': 'meat_on_bone',
        'sentiment_score': 0.129,
        'emoji_tkn': ':neutral:'
        },
    '\\u2668': {
        'meaning': 'hot_springs',
        'sentiment_score': 0.871,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3b9': {
        'meaning': 'musical_keyboard',
        'sentiment_score': 0.355,
        'emoji_tkn': ':positive:'
        },
    '\\u2655': {
        'meaning': 'white_chess_queen',
        'sentiment_score': 0.387,
        'emoji_tkn': ':positive:'
        },
    '\\u2583': {
        'meaning': 'lower_three_eighths_block',
        'sentiment_score': 0.903,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f698': {
        'meaning': 'oncoming_automobile',
        'sentiment_score': 0.067,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f34f': {
        'meaning': 'green_apple',
        'sentiment_score': 0.067,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f469': {
        'meaning': 'woman',
        'sentiment_score': 0.067,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f466': {
        'meaning': 'boy',
        'sentiment_score': 0.133,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f1ec': {
        'meaning': 'regional_indicator_symbol_letter_g',
        'sentiment_score': 0.267,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f1e7': {
        'meaning': 'regional_indicator_symbol_letter_b',
        'sentiment_score': 0.267,
        'emoji_tkn': ':positive:'
        },
    '\\u2620': {
        'meaning': 'skull_and_crossbones',
        'sentiment_score': -0.033,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f420': {
        'meaning': 'tropical_fish',
        'sentiment_score': 0.414,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f6b9': {
        'meaning': 'mens_symbol',
        'sentiment_score': 0.69,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4b5': {
        'meaning': 'banknote_with_dollar_sign',
        'sentiment_score': 0.379,
        'emoji_tkn': ':positive:'
        },
    '\\u2730': {
        'meaning': 'shadowed_white_star',
        'sentiment_score': 0.793,
        'emoji_tkn': ':positive:'
        },
    '\\u2560': {
        'meaning': 'box_drawings_double_vertical_and_right',
        'sentiment_score': 0.207,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f45b': {
        'meaning': 'purse',
        'sentiment_score': 0.357,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f699': {
        'meaning': 'recreational_vehicle',
        'sentiment_score': 0.036,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f331': {
        'meaning': 'seedling',
        'sentiment_score': 0.571,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4bb': {
        'meaning': 'personal_computer',
        'sentiment_score': 0.25,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f30f': {
        'meaning': 'earth_globe_asia-australia',
        'sentiment_score': 0.321,
        'emoji_tkn': ':positive:'
        },
    '\\u2584': {
        'meaning': 'lower_half_block',
        'sentiment_score': -0.071,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f453': {
        'meaning': 'eyeglasses',
        'sentiment_score': 0.296,
        'emoji_tkn': ':neutral:'
        },
    '\\u25c4': {
        'meaning': 'black_left-pointing_pointer',
        'sentiment_score': 0.222,
        'emoji_tkn': ':neutral:'
        },
    '\\u26be': {
        'meaning': 'baseball',
        'sentiment_score': -0.037,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f332': {
        'meaning': 'evergreen_tree',
        'sentiment_score': 0.385,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f474': {
        'meaning': 'older_man',
        'sentiment_score': 0.231,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3e0': {
        'meaning': 'house_building',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f347': {
        'meaning': 'grapes',
        'sentiment_score': 0.269,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f358': {
        'meaning': 'rice_cracker',
        'sentiment_score': 0.385,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f35b': {
        'meaning': 'curry_and_rice',
        'sentiment_score': 0.038,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f407': {
        'meaning': 'rabbit',
        'sentiment_score': 0.231,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f51e': {
        'meaning': 'no_one_under_eighteen_symbol',
        'sentiment_score': -0.038,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f475': {
        'meaning': 'older_woman',
        'sentiment_score': 0.423,
        'emoji_tkn': ':positive:'
        },
    '\\u25c0': {
        'meaning': 'black_left-pointing_triangle',
        'sentiment_score': 0.269,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f519': {
        'meaning': 'back_with_leftwards_arrow_above',
        'sentiment_score': 0.192,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f335': {
        'meaning': 'cactus',
        'sentiment_score': 0.192,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f43d': {
        'meaning': 'pig_nose',
        'sentiment_score': 0.12,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f36e': {
        'meaning': 'custard',
        'sentiment_score': -0.12,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f387': {
        'meaning': 'firework_sparkler',
        'sentiment_score': 0.68,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f40e': {
        'meaning': 'horse',
        'sentiment_score': 0.4,
        'emoji_tkn': ':positive:'
        },
    '\\u2794': {
        'meaning': 'heavy_wide-headed_rightwards_arrow',
        'sentiment_score': -0.12,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4b6': {
        'meaning': 'banknote_with_euro_sign',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f424': {
        'meaning': 'baby_chick',
        'sentiment_score': 0.52,
        'emoji_tkn': ':positive:'
        },
    '\\u2569': {
        'meaning': 'box_drawings_double_up_and_horizontal',
        'sentiment_score': 0.2,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f6c0': {
        'meaning': 'bath',
        'sentiment_score': 0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f311': {
        'meaning': 'new_moon_symbol',
        'sentiment_score': 0.458,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f6b2': {
        'meaning': 'bicycle',
        'sentiment_score': 0.375,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f411': {
        'meaning': 'sheep',
        'sentiment_score': -0.167,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3c1': {
        'meaning': 'chequered_flag',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f35e': {
        'meaning': 'bread',
        'sentiment_score': 0.042,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3be': {
        'meaning': 'tennis_racquet_and_ball',
        'sentiment_score': 0.542,
        'emoji_tkn': ':positive:'
        },
    '\\u255a': {
        'meaning': 'box_drawings_double_up_and_right',
        'sentiment_score': 0.292,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f239': {
        'meaning': 'squared_cjk_unified_ideograph-5272',
        'sentiment_score': 0.292,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f433': {
        'meaning': 'spouting_whale',
        'sentiment_score': 0.13,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f46e': {
        'meaning': 'police_officer',
        'sentiment_score': -0.348,
        'emoji_tkn': ':negative:'
        },
    '\\u2639': {
        'meaning': 'white_frowning_face',
        'sentiment_score': -0.522,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f435': {
        'meaning': 'monkey_face',
        'sentiment_score': 0.478,
        'emoji_tkn': ':positive:'
        },
    '\\u272a': {
        'meaning': 'circled_white_star',
        'sentiment_score': 0.304,
        'emoji_tkn': ':neutral:'
        },
    '\\u25d5': {
        'meaning': 'circle_with_all_but_upper_left_quadrant_black',
        'sentiment_score': 0.435,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f5fc': {
        'meaning': 'tokyo_tower',
        'sentiment_score': 0.522,
        'emoji_tkn': ':positive:'
        },
    '\\u2590': {
        'meaning': 'right_half_block',
        'sentiment_score': -0.13,
        'emoji_tkn': ':neutral:'
        },
    '\\u2660': {
        'meaning': 'black_spade_suit',
        'sentiment_score': 0.304,
        'emoji_tkn': ':neutral:'
        },
    '\\u2533': {
        'meaning': 'box_drawings_heavy_down_and_horizontal',
        'sentiment_score': -0.348,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f47a': {
        'meaning': 'japanese_goblin',
        'sentiment_score': -0.182,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f41a': {
        'meaning': 'spiral_shell',
        'sentiment_score': 0.227,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f442': {
        'meaning': 'ear',
        'sentiment_score': -0.136,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f5fd': {
        'meaning': 'statue_of_liberty',
        'sentiment_score': 0.318,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f375': {
        'meaning': 'teacup_without_handle',
        'sentiment_score': 0.364,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f192': {
        'meaning': 'squared_cool',
        'sentiment_score': 0.364,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f36f': {
        'meaning': 'honey_pot',
        'sentiment_score': 0.045,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f43a': {
        'meaning': 'wolf_face',
        'sentiment_score': 0.227,
        'emoji_tkn': ':neutral:'
        },
    '\\u21e8': {
        'meaning': 'rightwards_white_arrow',
        'sentiment_score': 0.455,
        'emoji_tkn': ':positive:'
        },
    '\\u27a8': {
        'meaning': 'heavy_concave-pointed_black_rightwards_arrow',
        'sentiment_score': 0.136,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f313': {
        'meaning': 'first_quarter_moon_symbol',
        'sentiment_score': 0.591,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f512': {
        'meaning': 'lock',
        'sentiment_score': 0.182,
        'emoji_tkn': ':positive:'
        },
    '\\u256c': {
        'meaning': 'box_drawings_double_vertical_and_horizontal',
        'sentiment_score': -0.136,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f473': {
        'meaning': 'man_with_turban',
        'sentiment_score': 0.619,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f302': {
        'meaning': 'closed_umbrella',
        'sentiment_score': 0.238,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f68c': {
        'meaning': 'bus',
        'sentiment_score': 0.19,
        'emoji_tkn': ':neutral:'
        },
    '\\u2669': {
        'meaning': 'quarter_note',
        'sentiment_score': 0.524,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f361': {
        'meaning': 'dango',
        'sentiment_score': -0.048,
        'emoji_tkn': ':neutral:'
        },
    '\\u2765': {
        'meaning': 'rotated_heavy_black_heart_bullet',
        'sentiment_score': 0.238,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3a1': {
        'meaning': 'ferris_wheel',
        'sentiment_score': 0.286,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f48c': {
        'meaning': 'love_letter',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f429': {
        'meaning': 'poodle',
        'sentiment_score': 0.35,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f31c': {
        'meaning': 'last_quarter_moon_with_face',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\u231a': {
        'meaning': 'watch',
        'sentiment_score': 0.2,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f6bf': {
        'meaning': 'shower',
        'sentiment_score': 0.6,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f416': {
        'meaning': 'pig',
        'sentiment_score': 0.15,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f506': {
        'meaning': 'high_brightness_symbol',
        'sentiment_score': 0.55,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f31b': {
        'meaning': 'first_quarter_moon_with_face',
        'sentiment_score': 0.55,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f482': {
        'meaning': 'guardsman',
        'sentiment_score': -0.15,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f414': {
        'meaning': 'chicken',
        'sentiment_score': 0.3,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f64e': {
        'meaning': 'person_with_pouting_face',
        'sentiment_score': -0.05,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3e9': {
        'meaning': 'love_hotel',
        'sentiment_score': 0.421,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f1eb': {
        'meaning': 'regional_indicator_symbol_letter_f',
        'sentiment_score': 0.474,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f528': {
        'meaning': 'hammer',
        'sentiment_score': -0.105,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4e2': {
        'meaning': 'public_address_loudspeaker',
        'sentiment_score': 0.421,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f426': {
        'meaning': 'bird',
        'sentiment_score': 0.421,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f432': {
        'meaning': 'dragon_face',
        'sentiment_score': -0.053,
        'emoji_tkn': ':neutral:'
        },
    '\\u267b': {
        'meaning': 'black_universal_recycling_symbol',
        'sentiment_score': 0.474,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f318': {
        'meaning': 'waning_crescent_moon_symbol',
        'sentiment_score': 0.579,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f350': {
        'meaning': 'pear',
        'sentiment_score': 0.158,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f314': {
        'meaning': 'waxing_gibbous_moon_symbol',
        'sentiment_score': 0.579,
        'emoji_tkn': ':positive:'
        },
    '\\u2565': {
        'meaning': 'box_drawings_down_double_and_horizontal_single',
        'sentiment_score': 0.105,
        'emoji_tkn': ':positive:'
        },
    '\\u274a': {
        'meaning': 'eight_teardrop-spoked_propeller_asterisk',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f456': {
        'meaning': 'jeans',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f6ba': {
        'meaning': 'womens_symbol',
        'sentiment_score': 0.167,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f617': {
        'meaning': 'kissing_face',
        'sentiment_score': 0.611,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3ad': {
        'meaning': 'performing_arts',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f404': {
        'meaning': 'cow',
        'sentiment_score': 0.278,
        'emoji_tkn': ':positive:'
        },
    '\\u25df': {
        'meaning': 'lower_left_quadrant_circular_arc',
        'sentiment_score': -0.056,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f362': {
        'meaning': 'oden',
        'sentiment_score': -0.111,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3a8': {
        'meaning': 'artist_palette',
        'sentiment_score': 0.167,
        'emoji_tkn': ':neutral:'
        },
    '\\u2b07': {
        'meaning': 'downwards_black_arrow',
        'sentiment_score': 0.389,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f6bc': {
        'meaning': 'baby_symbol',
        'sentiment_score': 0.556,
        'emoji_tkn': ':positive:'
        },
    '\\u26f2': {
        'meaning': 'fountain',
        'sentiment_score': 0.056,
        'emoji_tkn': ':neutral:'
        },
    '\\u2581': {
        'meaning': 'lower_one_eighth_block',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f1f4': {
        'meaning': 'regional_indicator_symbol_letter_o',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f317': {
        'meaning': 'last_quarter_moon_symbol',
        'sentiment_score': 0.611,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f316': {
        'meaning': 'waning_gibbous_moon_symbol',
        'sentiment_score': 0.611,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f505': {
        'meaning': 'low_brightness_symbol',
        'sentiment_score': 0.833,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f45c': {
        'meaning': 'handbag',
        'sentiment_score': 0.235,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f40c': {
        'meaning': 'snail',
        'sentiment_score': 0.647,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4bc': {
        'meaning': 'briefcase',
        'sentiment_score': 0.529,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f695': {
        'meaning': 'taxi',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f439': {
        'meaning': 'hamster_face',
        'sentiment_score': 0.294,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f320': {
        'meaning': 'shooting_star',
        'sentiment_score': 0.529,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f408': {
        'meaning': 'cat',
        'sentiment_score': 0.294,
        'emoji_tkn': ':neutral:'
        },
    '\\u21e7': {
        'meaning': 'upwards_white_arrow',
        'sentiment_score': 0.118,
        'emoji_tkn': ':neutral:'
        },
    '\\u260e': {
        'meaning': 'black_telephone',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f301': {
        'meaning': 'foggy',
        'sentiment_score': 0.176,
        'emoji_tkn': ':neutral:'
        },
    '\\u26ab': {
        'meaning': 'medium_black_circle',
        'sentiment_score': 0.235,
        'emoji_tkn': ':positive:'
        },
    '\\u2667': {
        'meaning': 'white_club_suit',
        'sentiment_score': 0.471,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3f0': {
        'meaning': 'european_castle',
        'sentiment_score': 0.294,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f6b5': {
        'meaning': 'mountain_bicyclist',
        'sentiment_score': 0.353,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3a2': {
        'meaning': 'roller_coaster',
        'sentiment_score': 0.471,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3b7': {
        'meaning': 'saxophone',
        'sentiment_score': 0.647,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f390': {
        'meaning': 'wind_chime',
        'sentiment_score': 0.176,
        'emoji_tkn': ':neutral:'
        },
    '\\u2508': {
        'meaning': 'box_drawings_light_quadruple_dash_horizontal',
        'sentiment_score': -0.588,
        'emoji_tkn': ':negative:'
        },
    '\\u2557': {
        'meaning': 'box_drawings_double_down_and_left',
        'sentiment_score': 0.353,
        'emoji_tkn': ':positive:'
        },
    '\\u2571': {
        'meaning': 'box_drawings_light_diagonal_upper_right_to_lower_left',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f307': {
        'meaning': 'sunset_over_buildings',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\u23f0': {
        'meaning': 'alarm_clock',
        'sentiment_score': 0.438,
        'emoji_tkn': ':positive:'
        },
    '\\u21e9': {
        'meaning': 'downwards_white_arrow',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f682': {
        'meaning': 'steam_locomotive',
        'sentiment_score': 0.25,
        'emoji_tkn': ':neutral:'
        },
    '\\u25e0': {
        'meaning': 'upper_half_circle',
        'sentiment_score': 0.438,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3bf': {
        'meaning': 'ski_and_ski_boot',
        'sentiment_score': 0.375,
        'emoji_tkn': ':positive:'
        },
    '\\u2726': {
        'meaning': 'black_four_pointed_star',
        'sentiment_score': 0.063,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f194': {
        'meaning': 'squared_id',
        'sentiment_score': 0.75,
        'emoji_tkn': ':positive:'
        },
    '\\u26ea': {
        'meaning': 'church',
        'sentiment_score': 0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f312': {
        'meaning': 'waxing_crescent_moon_symbol',
        'sentiment_score': 0.563,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f42a': {
        'meaning': 'dromedary_camel',
        'sentiment_score': 0.563,
        'emoji_tkn': ':positive:'
        },
    '\\u2554': {
        'meaning': 'box_drawings_double_down_and_right',
        'sentiment_score': 0.25,
        'emoji_tkn': ':neutral:'
        },
    '\\u255d': {
        'meaning': 'box_drawings_double_up_and_left',
        'sentiment_score': 0.438,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f454': {
        'meaning': 'necktie',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f531': {
        'meaning': 'trident_emblem',
        'sentiment_score': 0.067,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f193': {
        'meaning': 'squared_free',
        'sentiment_score': 0.2,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f40b': {
        'meaning': 'whale',
        'sentiment_score': 0.2,
        'emoji_tkn': ':positive:'
        },
    '\\u25bd': {
        'meaning': 'white_down-pointing_triangle',
        'sentiment_score': 0.4,
        'emoji_tkn': ':positive:'
        },
    '\\u2582': {
        'meaning': 'lower_one_quarter_block',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f41b': {
        'meaning': 'bug',
        'sentiment_score': 0.267,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f455': {
        'meaning': 't-shirt',
        'sentiment_score': 0.4,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f68b': {
        'meaning': 'tram_car',
        'sentiment_score': 0.067,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4b3': {
        'meaning': 'credit_card',
        'sentiment_score': 0.4,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f306': {
        'meaning': 'cityscape_at_dusk',
        'sentiment_score': 0.133,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3e7': {
        'meaning': 'automated_teller_machine',
        'sentiment_score': 0.8,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4a1': {
        'meaning': 'electric_light_bulb',
        'sentiment_score': 0.6,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f539': {
        'meaning': 'small_blue_diamond',
        'sentiment_score': 0.133,
        'emoji_tkn': ':neutral:'
        },
    '\\u2b05': {
        'meaning': 'leftwards_black_arrow',
        'sentiment_score': 0.467,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f360': {
        'meaning': 'roasted_sweet_potato',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f42b': {
        'meaning': 'bactrian_camel',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3ea': {
        'meaning': 'convenience_store',
        'sentiment_score': 0.067,
        'emoji_tkn': ':neutral:'
        },
    '\\u06e9': {
        'meaning': 'arabic_place_of_sajdah',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f1f1': {
        'meaning': 'regional_indicator_symbol_letter_l',
        'sentiment_score': 0.4,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4f9': {
        'meaning': 'video_camera',
        'sentiment_score': 0.429,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f45e': {
        'meaning': 'mans_shoe',
        'sentiment_score': 0.429,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f691': {
        'meaning': 'ambulance',
        'sentiment_score': 0.071,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f198': {
        'meaning': 'squared_sos',
        'sentiment_score': 0.071,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f45a': {
        'meaning': 'womans_clothes',
        'sentiment_score': 0.571,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f68d': {
        'meaning': 'oncoming_bus',
        'sentiment_score': 0.071,
        'emoji_tkn': ':neutral:'
        },
    '\\u25a1': {
        'meaning': 'white_square',
        'sentiment_score': -0.214,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f402': {
        'meaning': 'ox',
        'sentiment_score': 0.143,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f6a3': {
        'meaning': 'rowboat',
        'sentiment_score': 0.571,
        'emoji_tkn': ':positive:'
        },
    '\\u2733': {
        'meaning': 'eight_spoked_asterisk',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3c9': {
        'meaning': 'rugby_football',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f5fb': {
        'meaning': 'mount_fuji',
        'sentiment_score': 0.571,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f400': {
        'meaning': 'rat',
        'sentiment_score': 0.143,
        'emoji_tkn': ':neutral:'
        },
    '\\u2566': {
        'meaning': 'box_drawings_double_down_and_horizontal',
        'sentiment_score': 0.357,
        'emoji_tkn': ':positive:'
        },
    '\\u26fa': {
        'meaning': 'tent',
        'sentiment_score': 0.462,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f415': {
        'meaning': 'dog',
        'sentiment_score': 0.231,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3c2': {
        'meaning': 'snowboarder',
        'sentiment_score': 0.385,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f461': {
        'meaning': 'womans_sandal',
        'sentiment_score': 0.385,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4fb': {
        'meaning': 'radio',
        'sentiment_score': 0.308,
        'emoji_tkn': ':positive:'
        },
    '\\u2712': {
        'meaning': 'black_nib',
        'sentiment_score': 0.231,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f330': {
        'meaning': 'chestnut',
        'sentiment_score': 0.538,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3e2': {
        'meaning': 'office_building',
        'sentiment_score': 0.154,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f392': {
        'meaning': 'school_satchel',
        'sentiment_score': 0.462,
        'emoji_tkn': ':positive:'
        },
    '\\u2312': {
        'meaning': 'arc',
        'sentiment_score': 0.538,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3eb': {
        'meaning': 'school',
        'sentiment_score': -0.231,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f4f4': {
        'meaning': 'mobile_phone_off',
        'sentiment_score': 0.615,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f6a2': {
        'meaning': 'ship',
        'sentiment_score': 0.231,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f69a': {
        'meaning': 'delivery_truck',
        'sentiment_score': -0.077,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f409': {
        'meaning': 'dragon',
        'sentiment_score': 0.154,
        'emoji_tkn': ':neutral:'
        },
    '\\u2752': {
        'meaning': 'upper_right_shadowed_white_square',
        'sentiment_score': 0.231,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f40a': {
        'meaning': 'crocodile',
        'sentiment_score': 0.077,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f514': {
        'meaning': 'bell',
        'sentiment_score': 0.769,
        'emoji_tkn': ':positive:'
        },
    '\\u25e2': {
        'meaning': 'black_lower_right_triangle',
        'sentiment_score': 0.615,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3e5': {
        'meaning': 'hospital',
        'sentiment_score': 0.25,
        'emoji_tkn': ':positive:'
        },
    '\\u2754': {
        'meaning': 'white_question_mark_ornament',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f696': {
        'meaning': 'oncoming_taxi',
        'sentiment_score': -0.083,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f0cf': {
        'meaning': 'playing_card_black_joker',
        'sentiment_score': 0.083,
        'emoji_tkn': ':neutral:'
        },
    '\\u25bc': {
        'meaning': 'black_down-pointing_triangle',
        'sentiment_score': 0.083,
        'emoji_tkn': ':neutral:'
        },
    '\\u258c': {
        'meaning': 'left_half_block',
        'sentiment_score': -0.25,
        'emoji_tkn': ':negative:'
        },
    '\\u261b': {
        'meaning': 'black_right_pointing_index',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\u2729': {
        'meaning': 'stress_outlined_white_star',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f492': {
        'meaning': 'wedding',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f6a4': {
        'meaning': 'speedboat',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f410': {
        'meaning': 'goat',
        'sentiment_score': 0.417,
        'emoji_tkn': ':positive:'
        },
    '\\u25a0': {
        'meaning': 'black_square',
        'sentiment_score': -0.25,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f51a': {
        'meaning': 'end_with_leftwards_arrow_above',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3bb': {
        'meaning': 'violin',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f537': {
        'meaning': 'large_blue_diamond',
        'sentiment_score': 0.167,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f6a6': {
        'meaning': 'vertical_traffic_light',
        'sentiment_score': 0.083,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f513': {
        'meaning': 'open_lock',
        'sentiment_score': 0.083,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3bd': {
        'meaning': 'running_shirt_with_sash',
        'sentiment_score': 0.417,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4c5': {
        'meaning': 'calendar',
        'sentiment_score': 0.167,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3ba': {
        'meaning': 'trumpet',
        'sentiment_score': 0.583,
        'emoji_tkn': ':positive:'
        },
    '\\u272f': {
        'meaning': 'pinwheel_star',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f348': {
        'meaning': 'melon',
        'sentiment_score': -0.333,
        'emoji_tkn': ':negative:'
        },
    '\\u2709': {
        'meaning': 'envelope',
        'sentiment_score': 0.25,
        'emoji_tkn': ':positive:'
        },
    '\\u2563': {
        'meaning': 'box_drawings_double_vertical_and_left',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\u25e4': {
        'meaning': 'black_upper_left_triangle',
        'sentiment_score': 0.75,
        'emoji_tkn': ':positive:'
        },
    '\\u25cb': {
        'meaning': 'white_circle',
        'sentiment_score': 0.455,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f37c': {
        'meaning': 'baby_bottle',
        'sentiment_score': 0.455,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4c0': {
        'meaning': 'dvd',
        'sentiment_score': 0.091,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f69b': {
        'meaning': 'articulated_lorry',
        'sentiment_score': -0.182,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4d3': {
        'meaning': 'notebook',
        'sentiment_score': 0.182,
        'emoji_tkn': ':neutral:'
        },
    '\\u2609': {
        'meaning': 'sun',
        'sentiment_score': 0.182,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4b4': {
        'meaning': 'banknote_with_yen_sign',
        'sentiment_score': -0.182,
        'emoji_tkn': ':neutral:'
        },
    '\\u253c': {
        'meaning': 'box_drawings_light_vertical_and_horizontal',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f403': {
        'meaning': 'water_buffalo',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\u27b0': {
        'meaning': 'curly_loop',
        'sentiment_score': -0.091,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f50c': {
        'meaning': 'electric_plug',
        'sentiment_score': -0.091,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f344': {
        'meaning': 'mushroom',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4d5': {
        'meaning': 'closed_book',
        'sentiment_score': 0.182,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4e3': {
        'meaning': 'cheering_megaphone',
        'sentiment_score': 0.364,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f693': {
        'meaning': 'police_car',
        'sentiment_score': 0.273,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f417': {
        'meaning': 'boar',
        'sentiment_score': 0.455,
        'emoji_tkn': ':positive:'
        },
    '\\u21aa': {
        'meaning': 'rightwards_arrow_with_hook',
        'sentiment_score': 0.091,
        'emoji_tkn': ':neutral:'
        },
    '\\u26f3': {
        'meaning': 'flag_in_hole',
        'sentiment_score': 0.636,
        'emoji_tkn': ':positive:'
        },
    '\\u253b': {
        'meaning': 'box_drawings_heavy_up_and_horizontal',
        'sentiment_score': -0.364,
        'emoji_tkn': ':negative:'
        },
    '\\u251b': {
        'meaning': 'box_drawings_heavy_up_and_left',
        'sentiment_score': 0.545,
        'emoji_tkn': ':positive:'
        },
    '\\u2503': {
        'meaning': 'box_drawings_heavy_vertical',
        'sentiment_score': 0.364,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f471': {
        'meaning': 'person_with_blond_hair',
        'sentiment_score': 0.1,
        'emoji_tkn': ':neutral:'
        },
    '\\u23f3': {
        'meaning': 'hourglass_with_flowing_sand',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4ba': {
        'meaning': 'seat',
        'sentiment_score': 0.2,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3c7': {
        'meaning': 'horse_racing',
        'sentiment_score': -0.1,
        'emoji_tkn': ':neutral:'
        },
    '\\u263b': {
        'meaning': 'black_smiling_face',
        'sentiment_score': 0.2,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4de': {
        'meaning': 'telephone_receiver',
        'sentiment_score': 0.4,
        'emoji_tkn': ':positive:'
        },
    '\\u24b6': {
        'meaning': 'circled_latin_capital_letter_a',
        'sentiment_score': -0.1,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f309': {
        'meaning': 'bridge_at_night',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f6a9': {
        'meaning': 'triangular_flag_on_post',
        'sentiment_score': -0.2,
        'emoji_tkn': ':negative:'
        },
    '\\u270e': {
        'meaning': 'lower_right_pencil',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4c3': {
        'meaning': 'page_with_curl',
        'sentiment_score': 0.4,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3e8': {
        'meaning': 'hotel',
        'sentiment_score': 0.2,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4cc': {
        'meaning': 'pushpin',
        'sentiment_score': -0.4,
        'emoji_tkn': ':negative:'
        },
    '\\u264e': {
        'meaning': 'libra',
        'sentiment_score': -0.1,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4b7': {
        'meaning': 'banknote_with_pound_sign',
        'sentiment_score': 0.4,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f684': {
        'meaning': 'high-speed_train',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\u25b2': {
        'meaning': 'black_up-pointing_triangle',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\u26f5': {
        'meaning': 'sailboat',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f538': {
        'meaning': 'small_orange_diamond',
        'sentiment_score': 0.2,
        'emoji_tkn': ':neutral:'
        },
    '\\u231b': {
        'meaning': 'hourglass',
        'sentiment_score': 0.1,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f69c': {
        'meaning': 'tractor',
        'sentiment_score': 0.7,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f406': {
        'meaning': 'leopard',
        'sentiment_score': 0.3,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f452': {
        'meaning': 'womans_hat',
        'sentiment_score': 0.2,
        'emoji_tkn': ':positive:'
        },
    '\\u2755': {
        'meaning': 'white_exclamation_mark_ornament',
        'sentiment_score': 0.2,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f51b': {
        'meaning': 'on_with_exclamation_mark_with_left_right_arrow_above',
        'sentiment_score': 0.4,
        'emoji_tkn': ':positive:'
        },
    '\\u2662': {
        'meaning': 'white_diamond_suit',
        'sentiment_score': 0.3,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f1f2': {
        'meaning': 'regional_indicator_symbol_letter_m',
        'sentiment_score': 0.4,
        'emoji_tkn': ':positive:'
        },
    '\\u2745': {
        'meaning': 'tight_trifoliate_snowflake',
        'sentiment_score': 0.6,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f45d': {
        'meaning': 'pouch',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\u271e': {
        'meaning': 'shadowed_white_latin_cross',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\u25e1': {
        'meaning': 'lower_half_circle',
        'sentiment_score': 0.222,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f38b': {
        'meaning': 'tanabata_tree',
        'sentiment_score': 0.444,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f465': {
        'meaning': 'busts_in_silhouette',
        'sentiment_score': 0.222,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4f5': {
        'meaning': 'no_mobile_phones',
        'sentiment_score': 0.111,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f421': {
        'meaning': 'blowfish',
        'sentiment_score': 0.222,
        'emoji_tkn': ':neutral:'
        },
    '\\u25c6': {
        'meaning': 'black_diamond',
        'sentiment_score': 0.556,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3ef': {
        'meaning': 'japanese_castle',
        'sentiment_score': 0.111,
        'emoji_tkn': ':neutral:'
        },
    '\\u2602': {
        'meaning': 'umbrella',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f52d': {
        'meaning': 'telescope',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f3aa': {
        'meaning': 'circus_tent',
        'sentiment_score': 0.222,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f41c': {
        'meaning': 'ant',
        'sentiment_score': 0.444,
        'emoji_tkn': ':positive:'
        },
    '\\u264c': {
        'meaning': 'leo',
        'sentiment_score': 0.556,
        'emoji_tkn': ':positive:'
        },
    '\\u2610': {
        'meaning': 'ballot_box',
        'sentiment_score': -0.667,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f477': {
        'meaning': 'construction_worker',
        'sentiment_score': 0.222,
        'emoji_tkn': ':positive:'
        },
    '\\u21b3': {
        'meaning': 'downwards_arrow_with_tip_rightwards',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f508': {
        'meaning': 'speaker',
        'sentiment_score': 0.222,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4c4': {
        'meaning': 'page_facing_up',
        'sentiment_score': 0.667,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4cd': {
        'meaning': 'round_pushpin',
        'sentiment_score': 0.111,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f690': {
        'meaning': 'minibus',
        'sentiment_score': 0.556,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f694': {
        'meaning': 'oncoming_police_car',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f30b': {
        'meaning': 'volcano',
        'sentiment_score': 0.444,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4e1': {
        'meaning': 'satellite_antenna',
        'sentiment_score': 0.222,
        'emoji_tkn': ':positive:'
        },
    '\\u23e9': {
        'meaning': 'black_right-pointing_double_triangle',
        'sentiment_score': 0.111,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f6b3': {
        'meaning': 'no_bicycles',
        'sentiment_score': 0.667,
        'emoji_tkn': ':positive:'
        },
    '\\u2718': {
        'meaning': 'heavy_ballot_x',
        'sentiment_score': 0.556,
        'emoji_tkn': ':positive:'
        },
    '\\u06de': {
        'meaning': 'arabic_start_of_rub_el_hizb',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\u263e': {
        'meaning': 'last_quarter_moon',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f170': {
        'meaning': 'negative_squared_latin_capital_letter_a',
        'sentiment_score': 0.222,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4e5': {
        'meaning': 'inbox_tray',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f1fc': {
        'meaning': 'regional_indicator_symbol_letter_w',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\u2513': {
        'meaning': 'box_drawings_heavy_down_and_left',
        'sentiment_score': 0.444,
        'emoji_tkn': ':positive:'
        },
    '\\u2523': {
        'meaning': 'box_drawings_heavy_vertical_and_right',
        'sentiment_score': 0.444,
        'emoji_tkn': ':positive:'
        },
    '\\u24c1': {
        'meaning': 'circled_latin_capital_letter_l',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\u24ba': {
        'meaning': 'circled_latin_capital_letter_e',
        'sentiment_score': 0.333,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f526': {
        'meaning': 'electric_torch',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f464': {
        'meaning': 'bust_in_silhouette',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f681': {
        'meaning': 'helicopter',
        'sentiment_score': 0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3a0': {
        'meaning': 'carousel_horse',
        'sentiment_score': 0.375,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f401': {
        'meaning': 'mouse',
        'sentiment_score': -0.25,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4d7': {
        'meaning': 'green_book',
        'sentiment_score': 0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\u2510': {
        'meaning': 'box_drawings_light_down_and_left',
        'sentiment_score': -0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\u262e': {
        'meaning': 'peace_symbol',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\u2642': {
        'meaning': 'male_sign',
        'sentiment_score': 0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\u25de': {
        'meaning': 'lower_right_quadrant_circular_arc',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4ef': {
        'meaning': 'postal_horn',
        'sentiment_score': -0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f529': {
        'meaning': 'nut_and_bolt',
        'sentiment_score': 0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f462': {
        'meaning': 'womans_boots',
        'sentiment_score': 0.5,
        'emoji_tkn': ':positive:'
        },
    '\\u25c2': {
        'meaning': 'black_left-pointing_small_triangle',
        'sentiment_score': 0.25,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f4f0': {
        'meaning': 'newspaper',
        'sentiment_score': 0.25,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f4f6': {
        'meaning': 'antenna_with_bars',
        'sentiment_score': 0.25,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f6a5': {
        'meaning': 'horizontal_traffic_light',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f304': {
        'meaning': 'sunrise_over_mountains',
        'sentiment_score': 0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f5fe': {
        'meaning': 'silhouette_of_japan',
        'sentiment_score': 0.25,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f536': {
        'meaning': 'large_orange_diamond',
        'sentiment_score': 0.25,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3e4': {
        'meaning': 'european_post_office',
        'sentiment_score': 0.25,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f3a9': {
        'meaning': 'top_hat',
        'sentiment_score': 0.25,
        'emoji_tkn': ':neutral:'
        },
    '\\u24c2': {
        'meaning': 'circled_latin_capital_letter_m',
        'sentiment_score': 0.25,
        'emoji_tkn': ':positive:'
        },
    '\\U0001f527': {
        'meaning': 'wrench',
        'sentiment_score': -0.375,
        'emoji_tkn': ':negative:'
        },
    '\\U0001f405': {
        'meaning': 'tiger',
        'sentiment_score': 0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\u266e': {
        'meaning': 'music_natural_sign',
        'sentiment_score': 0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f17e': {
        'meaning': 'negative_squared_latin_capital_letter_o',
        'sentiment_score': -0.125,
        'emoji_tkn': ':neutral:'
        },
    '\\U0001f504': {
        'meaning': 'anticlockwise_downwards_and_upwards_open_circle_arrows',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\u2604': {
        'meaning': 'comet',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        },
    '\\u2628': {
        'meaning': 'cross_of_lorraine',
        'sentiment_score': 0.0,
        'emoji_tkn': ':neutral:'
        }
    }

CONTRACTIONS = { 
    r"ain't": "is not",
    r"aren't": "are not",
    r"can't": "cannot",
    r"can't've": "cannot have",
    r"'cause": "because",
    r"could've": "could have",
    r"couldn't": "could not",
    r"couldn't've": "could not have",
    r"didn't": "did not",
    r"doesn't": "does not",
    r"don't": "do not",
    r"hadn't": "had not",
    r"hadn't've": "had not have",
    r"hasn't": "has not",
    r"haven't": "have not",
    r"he'd": "he would",
    r"he'd've": "he would have",
    r"he'll": "he will",
    r"he'll've": "he will have",
    r"he's": "he is",
    r"how'd": "how did",
    r"how'd'y": "how do you",
    r"how'll": "how will",
    r"how're": "how are",
    r"how's": "how is",
    r"i'd": "i would",
    r"i'd've": "i would have",
    r"i'll": "i will",
    r"i'll've": "i will have",
    r"i'm": "i am",
    r"i've": "i have",
    r"isn't": "is not",
    r"it'd": "it would",
    r"it'd've": "it would have",
    r"it'll": "it will",
    r"it'll've": "it will have",
    r"it's": "it is",
    r"let's": "let us",
    r"ma'am": "madam",
    r"mayn't": "may not",
    r"might've": "might have",
    r"mightn't": "might not",
    r"mightn't've": "might not have",
    r"must've": "must have",
    r"mustn't": "must not",
    r"mustn't've": "must not have",
    r"needn't": "need not",
    r"needn't've": "need not have",
    r"o'clock": "of the clock",
    r"oughtn't": "ought not",
    r"oughtn't've": "ought not have",
    r"shan't": "shall not",
    r"sha'n't": "shall not",
    r"shan't've": "shall not have",
    r"she'd": "she would",
    r"she'd've": "she would have",
    r"she'll": "she will",
    r"she'll've": "she will have",
    r"she's": "she is",
    r"should've": "should have",
    r"shouldn't": "should not",
    r"shouldn't've": "should not have",
    r"so've": "so have",
    r"so's": "so is",
    r"that'd": "that would",
    r"that'd've": "that would have",
    r"that's": "that is",
    r"there'd": "there would",
    r"there'd've": "there would have",
    r"there's": "there is",
    r"they'd": "they would",
    r"they'd've": "they would have",
    r"they'll": "they will",
    r"they'll've": "they will have",
    r"they're": "they are",
    r"they've": "they have",
    r"'tis": "it is",
    r"to've": "to have",
    r"'twas": "it was",
    r"wasn't": "was not",
    r"we'd": "we would",
    r"we'd've": "we would have",
    r"we'll": "we will",
    r"we'll've": "we will have",
    r"we're": "we are",
    r"we've": "we have",
    r"weren't": "were not",
    r"what'll": "what will",
    r"what'll've": "what will have",
    r"what're": "what are",
    r"what's": "what is",
    r"what've": "what have",
    r"when's": "when is",
    r"when've": "when have",
    r"where'd": "where did",
    r"where's": "where is",
    r"where've": "where have",
    r"who'll": "who will",
    r"who'll've": "who will have",
    r"who's": "who is",
    r"who've": "who have",
    r"why's": "why is",
    r"why've": "why have",
    r"will've": "will have",
    r"won't": "will not",
    r"won't've": "will not have",
    r"would've": "would have",
    r"wouldn't": "would not",
    r"wouldn't've": "would not have",
    r"y'all'd've": "you all would have",
    r"y'all": "you all",
    r"y'all'd": "you all would",
    r"y'all're": "you all are",
    r"y'all've": "you all have",
    r"you'd": "you would",
    r"you'd've": "you would have",
    r"you'll": "you will",
    r"you'll've": "you will have",
    r"you're": "you are",
    r"you've": "you have"
    }