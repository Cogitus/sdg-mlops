import os
import re
import glob
import time
from unicodedata import normalize, combining

# For advanced NLP Processing
import spacy
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
# the rest of third party libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# models from spacy
import en_core_web_lg



if __name__ == '__main__':
    print('executado')