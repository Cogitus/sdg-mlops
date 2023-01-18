import argparse
import logging
import tempfile
from pathlib import Path
from typing import Optional

import wandb
from tqdm import tqdm

# for the typehint of the runs
from wandb.sdk.wandb_run import Run

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(
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


def main(args: argparse.Namespace) -> None:
    run = wandb.init(
        job_type="preprocess_data",
        project="sdg-onu",
        tags=["dev", "data", "preprocess"],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        # retrieving the data from weights and bias
        titles_path = load_data(
            run=run, artifact_name=args.titles_tag, local_savepath=Path(tmp_dir)
        )
        authors_path = load_data(
            run=run, artifact_name=args.authors_tag, local_savepath=Path(tmp_dir)
        )
        affiliations_path = load_data(
            run=run, artifact_name=args.affiliations_tag, local_savepath=Path(tmp_dir)
        )
        dois_path = load_data(
            run=run, artifact_name=args.dois_tag, local_savepath=Path(tmp_dir)
        )
        keywords_path = load_data(
            run=run, artifact_name=args.keywords_tag, local_savepath=Path(tmp_dir)
        )
        abstracts_path = load_data(
            run=run, artifact_name=args.abstracts_tag, local_savepath=Path(tmp_dir)
        )

        logging.info(titles_path)
        logging.info(authors_path)
        logging.info(affiliations_path)
        logging.info(dois_path)
        logging.info(keywords_path)
        logging.info(abstracts_path)


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
