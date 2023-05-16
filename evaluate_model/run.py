import argparse
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import nltk
import tensorflow as tf
from utils.data import download_artifact
from utils.language_processing import advanced_preprocess

import wandb

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args: argparse.Namespace) -> None:
    wandb_run = wandb.init(
        job_type="evaluate_model",
        project="sdg-onu",
        tags=[
            "spacy",
            "evaluate",
            "word_cloud",
            "evaluate_model",
            "tensorflow",
            "predict",
            "nltk",
        ],
    )

    tmp_dir = TemporaryDirectory()
    PATH_TMP = Path(tmp_dir.name)

    # Set the location of the NLTK data directory to a custom folder
    NLTK_DATA_FOLDER = PATH_TMP / "nltk_data"

    logger.info(f"starting download of nltk_data to {NLTK_DATA_FOLDER}")

    # without this, nltk won't know where to search the stopwords
    nltk.data.path.append(NLTK_DATA_FOLDER)
    nltk.download("punkt", download_dir=NLTK_DATA_FOLDER)
    nltk.download("stopwords", download_dir=NLTK_DATA_FOLDER)

    # CODE GOES HERE
    model_path = download_artifact(
        wandb_tag=args.model_tag,
        local_savepath=PATH_TMP / "model",
        wandb_run=wandb_run,
    )
    input_path = download_artifact(
        wandb_tag=args.input_data_tag,
        local_savepath=PATH_TMP,
        wandb_run=wandb_run,
        artifact_type="raw_data",
    )

    logger.info(f"MODEL_PATH = {model_path}")
    logger.info(f"INPUT_PATH = {input_path}")
    logger.info(os.listdir(PATH_TMP))

    # model = tf.keras.models.load_model()

    # CODE ENDS HERE

    tmp_dir.cleanup()
    wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download model and text to classify from wandb and visualize results"
    )

    parser.add_argument(
        "--model_tag",
        type=str,
        help="Path on W&B to the tensorflow model folder artifact",
        required=True,
    )

    parser.add_argument(
        "--input_data_tag",
        type=str,
        help="Path on W&B to the `titles_en.json` artifact",
        required=True,
    )

    ARGS = parser.parse_args()

    main(ARGS)
