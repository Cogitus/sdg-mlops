import logging
import time
from typing import Any

from googletrans import Translator
from httpcore._exceptions import ReadTimeout
from progressbar import ProgressBar

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def detect_langs(
    translator: Translator, text2convert: list[str], widgets: list[Any]
) -> list:
    """
    Detect the language of each text string in a given list of text.

    Args:
        translator (Translator): An instance of the Translator class to use for language detection.
        text2convert (list[str]): A list of text strings to detect the language of.
        widgets (list): A list of widget objects to display the progress bar.

    Returns:
        list: A list of detected languages corresponding to the input text strings.
    """
    MAX_SIZE = len(text2convert)
    logger.info(f"detecting languages for the input array of size {MAX_SIZE}")

    pbar = ProgressBar(maxval=MAX_SIZE, widgets=widgets, redirect_stdout=True).start()

    detected_langs = []
    for i, text in enumerate(text2convert):
        translation_success = False
        while not translation_success:
            try:
                translation = translator.detect(text)
                detected_langs.append(translation)
            except ReadTimeout:
                logger.warning(
                    f"failed to detect the language the {i}-th element. trying again"
                )
                time.sleep(0.75)
            else:
                translation_success = True
        pbar.update(i)
    pbar.finish()

    return [lang.lang for lang in detected_langs]


def translate_text(
    translator: Translator,
    text_arr: list[str],
    langs: list[str],
    widgets: list[Any],
    type_input: str,
    dest: str = "en",
    src: str = "pt",
) -> list[str]:
    """Translate a list of texts to a target language.

    Args:
        translator (Translator): An instance of the `Translator` class.
        text_arr (list[str]): A list of texts to be translated.
        langs (list[str]): A list of the languages of the texts to be translated.
        widgets (list[Any]): A list of progress bar widgets to display the progress
            of the translation.
        type_input (str): The type of the input texts. It can be "titles" or "keywords".
        dest (str, optional): The target language to translate the texts to. Defaults to "en".
        src (str, optional): The source language of the texts to be translated. Defaults to "pt".

    Returns:
        list[str]: A list of translated texts.
    """
    MAX_SIZE = len(text_arr)
    SLEEP_TIME = 1

    pbar = ProgressBar(maxval=MAX_SIZE, widgets=widgets).start()

    text_translated = []
    for i, lang, text in zip(range(MAX_SIZE), langs, text_arr):
        # if the input is `keywords` and not `titles` then type(text) == list
        if type_input == "keywords":
            text = ", ".join(text)
            SLEEP_TIME = 2

        if lang == "en":
            text_translated.append(text)
        if lang == "pt":
            translation_success = False
            while not translation_success:
                try:
                    text_translated.append(
                        translator.translate(text, dest=dest, src=src).text
                    )
                except ReadTimeout:
                    logger.warning(
                        f"failed to translate the {i}-th element. trying again"
                    )
                    # NOTE: it is only necessary to use sleep when there is a request
                    time.sleep(SLEEP_TIME)
                else:
                    translation_success = True
        pbar.update(i)
    pbar.finish()

    return text_translated
