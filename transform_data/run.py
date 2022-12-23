import time
import logging

import wandb
import progressbar
from googletrans import Translator



def detect_langs(translator, texts) -> list:
    widgets = [
        'Processing pages: ', progressbar.Percentage(),
        ' ', progressbar.Bar(), ' ', progressbar.ETA(),
        ' | ', progressbar.Counter(), ' Processed'
    ]
    pbar = progressbar.ProgressBar(maxval=len(titles), widgets=widgets).start()

    detected_langs = []
    for i, text in enumerate(texts):
        detected_langs.append(translator.detect(text))
        time.sleep(0.5)
        pbar.update(i)

    pbar.finish()

    langs = [lang.lang for lang in detected_langs]
    
    return langs


def translate_title(translator, titles, langs, dest="en", src="pt") -> list:
    widgets = [
        'Processing pages: ', progressbar.Percentage(),
        ' ', progressbar.Bar(), ' ', progressbar.ETA(), 
        ' | ', progressbar.Counter(), ' Processed'
    ]
    pbar = progressbar.ProgressBar(maxval=len(titles), widgets=widgets).start()

    titles_translated = []
    for i, lang, title in zip(range(len(titles)), langs, titles):
        if lang == "en":
            titles_translated.append(title)
        if lang == "pt":
            titles_translated.append(translator.translate(title, dest=dest, src=src).text)
            time.sleep(1)
        pbar.update(i)

    pbar.finish()
    
    return titles_translated


def translate_keywords(translator, keywords, langs, dest="en", src="pt") -> list:
    widgets = [
        'Processing pages: ', progressbar.Percentage(),
        ' ', progressbar.Bar(), ' ', progressbar.ETA(),
        ' | ', progressbar.Counter(), ' Processed'
    ]
    pbar = progressbar.ProgressBar(maxval=len(keywords), widgets=widgets).start()

    keywords_en = []
    for i, lang, keywords_list in zip(range(len(keywords)), langs, keywords):
        if lang == "en":
            keywords_en.append(", ".join(keywords_list))
        if lang == "pt":
            keywords_en.append(translator.translate(", ".join(keywords_list), dest=dest, src=src).text)
            time.sleep(2)
        pbar.update(i)

    pbar.finish()
    
    return keywords_en



if __name__ == '__main__':
    pass

    # translator = Translator()

    # langs = detect_langs(translator, titles)
    # titles_en = translate_title(translator, titles, langs)
    # keywords_en = translate_keywords(translator, keywords, langs)