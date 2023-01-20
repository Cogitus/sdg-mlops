import argparse
import logging
import tempfile
from functools import partial
from pathlib import Path

import wandb
from googletrans import Translator
from progressbar import ETA, Bar, Percentage, SimpleProgress, Timer
from utils.io import download_wandb_data, load_data, save_data
from utils.preprocessing import detect_langs, translate_text

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args: argparse.Namespace) -> None:
    run = wandb.init(
        job_type="preprocess_data",
        project="sdg-onu",
        tags=["dev", "data", "preprocess"],
    )

    translator = Translator()
    PBAR_WIDGETS = [
        "Processing pages: ",
        Percentage(),
        " (",
        SimpleProgress(),
        ") ",
        Bar(marker="●", fill="○"),
        " ",
        ETA(),
        " ",
        Timer(),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        # retrieving the data from W&B
        titles_path = download_wandb_data(
            run=run, artifact_name=args.titles_tag, local_savepath=Path(tmp_dir)
        )
        keywords_path = download_wandb_data(
            run=run, artifact_name=args.keywords_tag, local_savepath=Path(tmp_dir)
        )

        # this is the proper data at its proper variables
        titles = load_data(Path(tmp_dir, "titles.json"))
        keywords = load_data(Path(tmp_dir, "keywords.json"))

        # here we create the reference-language array
        langs = detect_langs(
            translator=translator, text2convert=titles, widgets=PBAR_WIDGETS
        )

        # differentiating between the two types of translations
        translate_titles = partial(
            translate_text, type_input="titles", widgets=PBAR_WIDGETS
        )
        translate_keywords = partial(
            translate_text, type_input="keywords", widgets=PBAR_WIDGETS
        )
        # the import to W&B is made by this
        wandb_save = partial(
            save_data,
            project_name="sdg-onu",
            run=run,
            tags_array=["dev", "data", "preprocess_data"],
            remote=True,
        )

        titles_en = translate_titles(
            translator=translator, text_arr=titles, langs=langs
        )
        wandb_save(data=titles_en, filename="titles_en.json")

        keywords_en = translate_keywords(
            translator=translator, text_arr=keywords, langs=langs
        )
        wandb_save(data=keywords_en, filename="keywords_en.json")

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file from Weights and Biases (W&B) and preprocess its text contents"
    )

    parser.add_argument(
        "--titles_tag",
        type=str,
        help="Path on W&B to the titles artifact",
        required=True,
    )
    parser.add_argument(
        "--authors_tag",
        type=str,
        help="Path on W&B to the authors artifact",
        required=True,
    )
    parser.add_argument(
        "--affiliations_tag",
        type=str,
        help="Path on W&B to the affiliations artifact",
        required=True,
    )
    parser.add_argument(
        "--dois_tag", type=str, help="Path on W&B to the dois artifact", required=True
    )
    parser.add_argument(
        "--keywords_tag",
        type=str,
        help="Path on W&B to the keywords artifact",
        required=True,
    )
    parser.add_argument(
        "--abstracts_tag",
        type=str,
        help="Path on W&B to the abstracts artifact",
        required=True,
    )
    args = parser.parse_args()

    main(args)
