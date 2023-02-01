import logging

import spacy

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:  # If not present, we download
        logger.error(
            "spacy model `en_core_web_lg` is already installed, start donwload"
        )
        spacy.cli.download("en_core_web_lg")
    else:
        logger.info(
            "spacy model `en_core_web_lg` is already installed, jumping to next step"
        )
