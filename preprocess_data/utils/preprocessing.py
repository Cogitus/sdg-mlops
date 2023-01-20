import logging
import time
from typing import Any, List

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
    translator: Translator, text2convert: List[str], widgets: List[Any]
) -> list:
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
    text_arr: List[str],
    langs: List[str],
    widgets: List[Any],
    type_input: str,
    dest: str = "en",
    src: str = "pt",
) -> List[str]:
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
