import logging
import re
from collections.abc import Iterable
from multiprocessing import cpu_count
from unicodedata import combining, normalize

import nltk
import numpy as np
import spacy
import tensorflow as tf
from joblib import Parallel, delayed
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_stopwords() -> list[str]:

    nltk_stopwords = nltk.corpus.stopwords.words("english")
    nltk_stopwords = set(nltk_stopwords)

    spacy_en = spacy.load("en_core_web_lg")
    spacy_stopwords = spacy_en.Defaults.stop_words

    # joining both set of stopwords
    stopwords = spacy_stopwords.union(nltk_stopwords)
    stopwords = list(stopwords)

    # tmp_dir.cleanup()

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

    # 6) Tokenize text (in parallel with joblib, and the backend is with multiprocessing
    # since with the default one it has problems with the nltk data downloaded
    # to a custom folder
    Z = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(word_tokenize)(text) for text in Z
    )

    # 7) Remove stopwords (in parallel with joblib)
    stopwords = get_stopwords()
    clean_tokens_func = lambda tokens: list(
        (word for word in tokens if ((word not in stopwords) and (len(word) > 1)))
    )
    Z = Parallel(n_jobs=-1)(delayed(clean_tokens_func)(tokens) for tokens in Z)

    # 8.1) Lemmatizing (speeding up it with nlp.pipe as suggested by the spacy documentation)
    if truncation == "lemma":
        # 8.1.1) Concatenate tokens
        Z = [" ".join(tokens) for tokens in Z]

        nlp = spacy.load("en_core_web_lg")

        # if all the cores are being used, there is a high chance of crashing
        NUM_CORES = cpu_count() - 1
        Z_lemma = []

        # 8.1.2) Lemmatize sentences
        for doc in nlp.pipe(Z, batch_size=128, n_process=NUM_CORES):
            Z_lemma.append(" ".join([tok.lemma_ for tok in doc]))
        Z = Z_lemma
    # 8.2) Stemming (in parallel with joblib)
    elif truncation == "stem":
        stemmer = SnowballStemmer("english")

        stemming_func = lambda tokens: " ".join(
            [stemmer.stem(token) for token in tokens]
        )
        Z = Parallel(n_jobs=-1)(delayed(stemming_func)(tokens) for tokens in Z)
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


def create_dataset(X: Iterable[str], y: Iterable[Iterable[float]]) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
