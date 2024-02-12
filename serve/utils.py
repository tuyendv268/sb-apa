import re
import pandas as pd

def load_lexicon(path):
    lexicon = pd.read_csv(path, names=["word", "arpa"], sep="\t")

    lexicon.dropna(inplace=True)
    lexicon["word"] = lexicon.word.apply(lambda x: x.lower())
    lexicon["arpa"] = lexicon.arpa.apply(lambda x: re.sub("\d", "", x).lower())

    lexicon.word.drop_duplicates(inplace=True, keep='last')
    lexicon.set_index("word", inplace=True)
    lexicon = lexicon.to_dict()["arpa"]

    return lexicon

def normalize(text):
    text = re.sub(
        r'[\!@#$%^&*\(\)\\\.\"\,\?\;\:\+\-\_\/\|~`]', ' ', text)
    
    text = re.sub('\s+', ' ', text)
    text = text.lower().strip()
    return text

