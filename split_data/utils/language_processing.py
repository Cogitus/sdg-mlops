import logging
import re
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from unicodedata import combining, normalize

import nltk
import numpy as np
import spacy
import tqdm
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_stopwords() -> list[str]:
    stopwords = None

    with TemporaryDirectory() as tmp_dir:
        logger.info("starting download of nltk_data")

        # Set the location of the NLTK data directory to a custom folder
        nltk_data_folder = Path(tmp_dir) / "nltk_data"

        # without this, nltk won't know where to search the stopwords
        nltk.data.path.append(nltk_data_folder)
        nltk.download("punkt", download_dir=nltk_data_folder)
        nltk.download("stopwords", download_dir=nltk_data_folder)

        nltk_stopwords = nltk.corpus.stopwords.words("english")
        nltk_stopwords = set(nltk_stopwords)

        spacy_en = spacy.load("en_core_web_lg")
        spacy_stopwords = spacy_en.Defaults.stop_words

        # joining both set of stopwords
        stopwords = spacy_stopwords.union(nltk_stopwords)
        stopwords = list(stopwords)

    return stopwords


def advanced_preprocess(
    X: Iterable[str], y: Iterable[Iterable[float]], truncation: str = "lemma"
) -> tuple[Iterable[str], Iterable[Iterable[float]]]:
    # 1) Convert text to lowercase
    Z = [text.lower() for text in X]

    # 2) Remove special characters
    SPECIAL_CHAR_REG_EX = "!@#$%^&*()[]{};:,./<>?\|`~-=_+123456789"
    translation_table = {ord(char): " " for char in SPECIAL_CHAR_REG_EX}
    Z = [text.translate(translation_table) for text in Z]

    # 3) Remove numbers (with a whitespace at its side or sides and being only the number)
    Z = [re.sub(r"^\d+\s|\s\d+\s|\s\d+$|\d+\)", " ", text) for text in Z]

    # 4) Remove double spaces
    Z = [re.sub(r"\s+[a-zA-Z]\s+", " ", text) for text in Z]

    # 5) Remove accents (normalize('NFKD', str) do things like Ç -> C + ̧)
    Z = [
        "".join([char for char in normalize("NFKD", text) if not combining(char)])
        for text in Z
    ]

    # 6) Tokenize text
    Z = [word_tokenize(text) for text in Z]

    # 7) Remove stopwords
    stopwords = get_stopwords()
    Z = [
        list((word for word in tokens if ((word not in stopwords) and (len(word) > 1))))
        for tokens in Z
    ]

    # 8.1) Lemmatizing
    if truncation == "lemma":
        # 8.1.1) Concatenate tokens
        Z = [" ".join(tokens) for tokens in Z]

        # 8.1.2) Lemmatize sentences
        nlp = spacy.load("en_core_web_lg")
        lemmatize = lambda sentence: " ".join([token.lemma_ for token in nlp(sentence)])
        Z = [lemmatize(text) for text in tqdm(Z)]
    # 8.2) Stemming
    elif truncation == "stem":
        stemmer = SnowballStemmer("english")
        Z = [" ".join([stemmer.stem(token) for token in tokens]) for tokens in Z]
    # 8.3) Truncation is None
    else:
        Z = [" ".join(tokens) for tokens in Z]

    # 9) Convert back to np.array
    Z = np.array(Z)

    # 10) Discard empty sentences
    non_empty_sentences = Z != ""
    y = y[non_empty_sentences]
    Z = Z[non_empty_sentences]

    return Z, y
