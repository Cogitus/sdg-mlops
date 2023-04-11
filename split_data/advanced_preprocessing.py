import argparse
import logging
from os import mkdir
from pathlib import Path
from tempfile import TemporaryDirectory

import nltk
import tensorflow as tf
import wandb
from utils.language_processing import advanced_preprocess, create_dataset
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
        tags=["dev", "data", "dataset", "split_data", "tensorflow", "split", "nltk"],
    )

    tmp_dir = TemporaryDirectory()
    path_tmp = Path(tmp_dir.name)

    # Set the location of the NLTK data directory to a custom folder
    nltk_data_folder = path_tmp / "nltk_data"

    logger.info(f"starting download of nltk_data to {nltk_data_folder}")

    # without this, nltk won't know where to search the stopwords
    nltk.data.path.append(nltk_data_folder)
    nltk.download("punkt", download_dir=nltk_data_folder)
    nltk.download("stopwords", download_dir=nltk_data_folder)

    X_train = download_wandb_data(args.X_train, run=run, local_savepath=path_tmp)
    y_train = download_wandb_data(args.y_train, run=run, local_savepath=path_tmp)
    X_valid = download_wandb_data(args.X_valid, run=run, local_savepath=path_tmp)
    y_valid = download_wandb_data(args.y_valid, run=run, local_savepath=path_tmp)
    X_test = download_wandb_data(args.X_test, run=run, local_savepath=path_tmp)
    y_test = download_wandb_data(args.y_test, run=run, local_savepath=path_tmp)

    # doing the proper NLP preprocessing here
    logger.info("Starting advanced preprocess for `train` splits.")
    X_train, y_train = advanced_preprocess(
        X_train.to_numpy().flatten(), y_train.to_numpy()
    )

    logger.info("Starting advanced preprocess for `validation` splits.")
    X_valid, y_valid = advanced_preprocess(
        X_valid.to_numpy().flatten(), y_valid.to_numpy()
    )

    logger.info("Starting advanced preprocess for `test` splits.")
    X_test, y_test = advanced_preprocess(X_test.to_numpy().flatten(), y_test.to_numpy())

    # build train set
    train_set = (
        create_dataset(X_train, y_train)
        .shuffle(X_train.shape[0], seed=args.tf_seed)
        .batch(args.tf_batch_size)
        .prefetch(1)
    )
    # build validation set
    valid_set = create_dataset(X_valid, y_valid).batch(args.tf_batch_size).prefetch(1)
    # build test set
    test_set = create_dataset(X_test, y_test).batch(args.tf_batch_size).prefetch(1)

    datasets_folder = path_tmp / "datasets_sdg_tensorflow"
    mkdir(datasets_folder)

    tf.data.Dataset.save(train_set, (datasets_folder / "train_set").as_posix())
    tf.data.Dataset.save(valid_set, (datasets_folder / "valid_set").as_posix())
    tf.data.Dataset.save(test_set, (datasets_folder / "test_set").as_posix())

    tensorflow_datasets = wandb.Artifact(
        name="tensorflow_datasets",
        type="dataset",
        description="""The folder that contains the training/validation/testing datasets
        to be used by Tensorflow""",
        metadata={"batch_size": args.tf_batch_size, "seed": args.tf_seed},
    )

    tensorflow_datasets.add_dir(local_path=datasets_folder)
    run.log_artifact(tensorflow_datasets)
    tensorflow_datasets.wait()

    tmp_dir.cleanup()
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
    parser.add_argument(
        "--tf_batch_size",
        type=int,
        help="Batch size collected of dataset after creating it with Tensorflow",
        required=True,
    )
    parser.add_argument(
        "--tf_seed",
        type=int,
        help="Seed used to shuffle the Tensorflow dataset",
        required=True,
    )
    ARGS = parser.parse_args()

    main(ARGS)
