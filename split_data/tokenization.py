import shutil
from pathlib import Path

import en_core_web_lg
import nltk
import spacy

# Set the location of the NLTK data directory to a custom folder
nltk_data_folder = Path.cwd() / "nltk_data"

nltk.download("punkt", download_dir=nltk_data_folder)
nltk.download("stopwords", download_dir=nltk_data_folder)


shutil.rmtree(nltk_data_folder)
