import en_core_web_lg
import spacy

nlp = en_core_web_lg.load()
doc = nlp(u"This is a sentence.")
