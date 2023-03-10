import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import wandb
from utils.operations import download_wandb_data

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# @hydra.main()
def main(args: argparse.Namespace) -> None:
    run = wandb.init(
        job_type="training_dataset_splitting",
        project="sdg-onu",
        tags=["dev", "data", "dataset", "split_data", "tensorflow", "split"],
    )

    with TemporaryDirectory() as tmp_dir:

        X_train_path = download_wandb_data(
            args.X_train, run=run, local_savepath=tmp_dir
        )
        y_train_path = download_wandb_data(
            args.y_train, run=run, local_savepath=tmp_dir
        )
        X_valid_path = download_wandb_data(
            args.X_valid, run=run, local_savepath=tmp_dir
        )
        y_valid_path = download_wandb_data(
            args.y_valid, run=run, local_savepath=tmp_dir
        )
        X_test_path = download_wandb_data(args.X_test, run=run, local_savepath=tmp_dir)
        y_test_path = download_wandb_data(args.y_test, run=run, local_savepath=tmp_dir)

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
