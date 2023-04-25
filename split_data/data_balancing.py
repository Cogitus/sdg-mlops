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
    data_loader = LocalSDGLoader()
    data_loader.load_data(data_location=args.path_sdg_dataset)

    data = data_loader.datasets

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
            metadata={
                "quantile": args.quantile,
                "random_state": args.random_state,
                "label_columns": label_columns,
            },
        )

        tmp_dir = TemporaryDirectory()

        csv_path = Path(tmp_dir.name) / "balanced_table.csv"
        balanced_data.to_csv(csv_path, index=False)

        balanced_table.add_file(local_path=csv_path)
        run.log_artifact(balanced_table)
        balanced_table.wait()

        tmp_dir.cleanup()


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
    parser.add_argument(
        "--path_sdg_dataset",
        type=Path,
        help="The random seed used to sample the dataframe",
        required=True,
    )

    ARGS = parser.parse_args()

    main(ARGS)
