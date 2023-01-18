import argparse
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import wandb
from googletrans import Translator
from progressbar import ETA, Bar, Percentage, ProgressBar, SimpleProgress, Timer

# for the typehint of the runs
from wandb.sdk.wandb_run import Run

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_wandb_data(
    artifact_name: str,
    run: Run,
    local_savepath: Optional[Path] = Path("./artifacts"),
) -> Path:
    # import artifact from wandb
    logger.info(
        f"loading the `{artifact_name}` from W&B to {local_savepath.absolute()}"
    )
    artifact = run.use_artifact(artifact_name, type="raw_data")
    local_savepath = artifact.download(local_savepath)

    return local_savepath


def load_data(filepath: Path) -> Any:
    with open(filepath, "r") as file:
        return json.load(file)


def detect_langs(translator: Translator, text2convert: list, widgets: list) -> list:
    pbar = ProgressBar(
        maxval=len(text2convert), widgets=widgets, redirect_stdout=True
    ).start()

    detected_langs = []
    for i, text in enumerate(text2convert):
        detected_langs.append(translator.detect(text))
        time.sleep(0.5)
        pbar.update(i)
    pbar.finish()

    return [lang.lang for lang in detected_langs]


# def translate_data(translator: Translator, data_arr: list, langs: list) -> list:
#     ...


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
        authors_path = download_wandb_data(
            run=run, artifact_name=args.authors_tag, local_savepath=Path(tmp_dir)
        )
        affiliations_path = download_wandb_data(
            run=run, artifact_name=args.affiliations_tag, local_savepath=Path(tmp_dir)
        )
        dois_path = download_wandb_data(
            run=run, artifact_name=args.dois_tag, local_savepath=Path(tmp_dir)
        )
        keywords_path = download_wandb_data(
            run=run, artifact_name=args.keywords_tag, local_savepath=Path(tmp_dir)
        )
        abstracts_path = download_wandb_data(
            run=run, artifact_name=args.abstracts_tag, local_savepath=Path(tmp_dir)
        )

        # this is the proper data at its proper variables
        titles = load_data(Path(tmp_dir, "titles.json"))
        authors = load_data(Path(tmp_dir, "authors.json"))
        affiliations = load_data(Path(tmp_dir, "affiliations.json"))
        dois = load_data(Path(tmp_dir, "dois.json"))
        keywords = load_data(Path(tmp_dir, "keywords.json"))
        abstracts = load_data(Path(tmp_dir, "abstracts.json"))

        # here we create the reference-language array
        langs = detect_langs(
            translator=translator, text2convert=titles, widgets=PBAR_WIDGETS
        )

        logger.info(f"type of langs unit: {type(langs[0])}")


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
