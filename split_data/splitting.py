import argparse
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(args: argparse.Namespace):
    run = wandb.init(
        job_type="training_dataset_splitting",
        project="sdg-onu",
        tags=["data", "preprocess", "splitting", "sklearn", "scikit", "train", "test"],
    )

    with TemporaryDirectory() as tmp_dir:
        # 1) import artifact from wandb
        logger.info(f"starting download of {args.dataset_name} from wandb server")

        artifact = run.use_artifact(args.dataset_name, type="dataset")
        local_savepath = artifact.download(tmp_dir)
        artifact_path = Path(local_savepath) / "balanced_data.table.json"

        logger.info(f"artifact `{args.dataset_name}` downloaded at {artifact_path}")

        with open(artifact_path, "rt") as file:
            downloaded_artifact = json.load(file)
            balanced_dataset = pd.DataFrame(
                downloaded_artifact["data"], columns=downloaded_artifact["columns"]
            )

        # 2) proper splitting of the dataset
        logger.info("splitting dataset into train/test data samples")

        SDG_COLUMNS = balanced_dataset.columns[1:].to_list()

        X_train, X_test, y_train, y_test = train_test_split(
            np.array(balanced_dataset["text"]),
            np.array(balanced_dataset[SDG_COLUMNS]),
            test_size=args.test_share_size,
            random_state=args.random_state,
        )

        # we'll create a validation split from X_train/y_train, so
        # train_size_factor is the proportion of training/validation data
        TRAIN_SIZE = round(args.train_size_factor * X_train.shape[0])

        # since there are too many dataset splits, to DRY the code, we'll be
        # unifying them on a iterable
        dataset_splits = (
            ("X_train", X_train),
            ("y_train", y_train),
            ("X_valid", X_train[TRAIN_SIZE:]),
            ("y_valid", y_train[TRAIN_SIZE:]),
            ("X_test", X_train[:TRAIN_SIZE]),
            ("y_test", y_train[:TRAIN_SIZE]),
        )

        # # 3) logging to the terminal
        logger.warning(f"train set: \t{X_train.shape[0]} records.")
        logger.warning(f"validation set: {X_train[TRAIN_SIZE:].shape[0]} records.")
        logger.warning(f"test set: \t{X_train[:TRAIN_SIZE].shape[0]} records.")

        # 4) saving dataset splits as W&B artifacts
        with wandb.init(
            job_type="splitting",
            project="sdg-onu",
            tags=["train", "test", "dataset", "preprocessing", "splitting"],
        ) as run:
            # make the limit of rows higher to comport the dataset
            wandb.Table.MAX_ARTIFACT_ROWS = 2_000_000

            for split_name, data in dataset_splits:
                logger.info(f"Logging `{split_name}` artifact")

                # saving the split as a .csv to be in a easier format for retrieval
                # OBS: the lambda function serves to give a different set of columns
                # if the data_split is of the type X_ (that only has one column,
                # that is 'text') or y_ (that has all the 16 sdg columns)
                csv_path = Path(tmp_dir) / (split_name + ".csv")
                pd.DataFrame(
                    data,
                    columns=(lambda arg: ["text"] if "X" in arg else SDG_COLUMNS)(
                        split_name
                    ),
                ).to_csv(csv_path)

                logger.debug("creating description of artifact")

                # formatted using (lambda x: output)(input) structure for inline processing
                artifact_description = (
                    "Split of the full dataset used as "
                    f"{(lambda arg: 'input' if 'X' in arg else 'label/output')(split_name)}"
                    " at "
                    f"{(lambda arg: 'training' if 'train' in arg else ('validation' if 'valid' in arg else 'testing'))(split_name)}"
                )

                logger.debug("creating artifact")
                artifact = wandb.Artifact(
                    name=split_name,
                    type="dataset",
                    description=artifact_description,
                    metadata={
                        "test_share_size": args.test_share_size,
                        "random_state": args.random_state,
                        "train_size_factor": args.train_size_factor,
                    },
                )

                logger.debug("adding artifacts finally")

                artifact.add_file(local_path=csv_path)

                logger.debug("logging the artifacts to W&B")
                run.log_artifact(artifact)

                artifact.wait()

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splits dataset in train/test batches")

    parser.add_argument(
        "--train_size_factor",
        type=float,
        help="Coeficient that tells the proportion of the training dataset",
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Path of the preprocessed/balanced dataset at W&B",
        required=True,
    )
    parser.add_argument(
        "--test_share_size",
        type=float,
        help="The number that defines the size of the test set (as a percentage of the total)",
        required=True,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="The random seed used to randomize the split of the full dataset.",
        required=True,
    )
    ARGS = parser.parse_args()

    main(ARGS)
