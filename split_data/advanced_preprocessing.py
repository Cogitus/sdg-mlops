import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import tensorflow as tf
import wandb
from utils.language_processing import advanced_preprocess
from utils.operations import download_wandb_data

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args: argparse.Namespace) -> None:
    run = wandb.init(
        job_type="training_dataset_splitting",
        project="sdg-onu",
        tags=["dev", "data", "dataset", "split_data", "tensorflow", "split"],
    )

    with TemporaryDirectory() as tmp_dir:
        X_train = download_wandb_data(args.X_train, run=run, local_savepath=tmp_dir)
        y_train = download_wandb_data(args.y_train, run=run, local_savepath=tmp_dir)
        X_valid = download_wandb_data(args.X_valid, run=run, local_savepath=tmp_dir)
        y_valid = download_wandb_data(args.y_valid, run=run, local_savepath=tmp_dir)
        X_test = download_wandb_data(args.X_test, run=run, local_savepath=tmp_dir)
        y_test = download_wandb_data(args.y_test, run=run, local_savepath=tmp_dir)

        # doing the proper preprocess here
        X_train, y_train = advanced_preprocess(X_train, y_train)
        X_valid, y_valid = advanced_preprocess(X_valid, y_valid)
        X_test, y_test = advanced_preprocess(X_test, y_test)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download preprocessed-artifacts from Weights and Biases (W&B) and create a TF dataset"
    )

    parser.add_argument(
        "--X_train",
        type=str,
        help="Path on W&B to the `X_train` artifact",
        required=True,
    )
    parser.add_argument(
        "--y_train",
        type=str,
        help="Path on W&B to the `y_train` artifact",
        required=True,
    )
    parser.add_argument(
        "--X_valid",
        type=str,
        help="Path on W&B to the `X_valid` artifact",
        required=True,
    )
    parser.add_argument(
        "--y_valid",
        type=str,
        help="Path on W&B to the `y_valid` artifact",
        required=True,
    )
    parser.add_argument(
        "--X_test",
        type=str,
        help="Path on W&B to the `X_test` artifact",
        required=True,
    )
    parser.add_argument(
        "--y_test",
        type=str,
        help="Path on W&B to the `y_test` artifact",
        required=True,
    )
    args = parser.parse_args()

    main()
