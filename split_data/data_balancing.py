import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import wandb
from utils.data_loader import LocalSDGLoader
from utils.operations import balance_multilabel_dataset

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args: argparse.Namespace) -> None:
    # current_dir = Path.cwd()

    data_loader = LocalSDGLoader()

    with TemporaryDirectory() as tmp_dir:
        data = data_loader.load_data(
            data_location=Path("/home/alsinaariel/Downloads/SDGs")
        )

        # preprocess the data by balancing it with a quantile of 0.5
        label_columns = [col for col in data.columns if col.startswith("SDG")]
        balanced_data = balance_multilabel_dataset(
            dataset=data,
            label_columns=label_columns,
            quantile=args.quantile,
            random_state=args.random_state,
        )

        with wandb.init(
            job_type="balancing_data",
            project="sdg-onu",
            tags=["quantile", "balancing", "multilabel", "preprocessing", "splitting"],
        ) as run:
            balanced_table = wandb.Artifact(
                name="balanced_table",
                type="dataset",
                description="The balanced dataset after the preprocessing of removel of duplicate lines and label balancing",
                metadata={"quantile": args.quantile, "random_state": args.random_state},
            )

            # make the limit of rows higher to comport the dataset
            wandb.Table.MAX_ARTIFACT_ROWS = 2_000_000

            balanced_table.add(
                obj=wandb.Table(dataframe=balanced_data), name="balanced_data"
            )

            logger.info(f"Logging `balanced_data` artifact.")

            run.log_artifact(balanced_table)
            balanced_table.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downloader of the full dataset and balancer of it"
    )

    parser.add_argument(
        "--quantile",
        type=float,
        help="Counts of labels at which to balance the dataset.",
        required=True,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="The random seed used to sample the dataframe",
        required=True,
    )
    ARGS = parser.parse_args()

    main(ARGS)
