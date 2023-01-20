import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import wandb

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
    logger.info(
        f"loading the `{artifact_name}` from W&B to {local_savepath.absolute()}"
    )

    # import artifact from wandb
    artifact = run.use_artifact(artifact_name, type="raw_data")
    local_savepath = artifact.download(local_savepath)

    return local_savepath


def load_data(filepath: Path) -> Any:
    with open(filepath, "r") as file:
        return json.load(file)


def save_data(
    data: Any,
    project_name: str,
    filename: str,
    tags_array: List[str],
    run: Run,
    filepath: Optional[str] = Path.cwd(),
    remote: Optional[bool] = False,
) -> None:
    # local where the file already is or where is to save it (locally)
    save_path = Path(filepath, filename)

    if remote:
        logger.info(f"Saving {filename} at a wandb project `{project_name}`")

        with tempfile.TemporaryDirectory() as TMP_DIR:
            logger.info("Starting conection with WandB")

            with wandb.init(
                job_type="download_data",
                project=project_name,
                tags=tags_array,
            ) as run:
                logger.info(f"Creating artifact for {filename} at {TMP_DIR}")

                # instatiating a new artifact for the data
                artifact = wandb.Artifact(
                    name=filename, type="raw_data", description=""
                )

                try:
                    # contents of the file for 0 is '' and 2 is '[]'
                    is_empty_file = os.path.getsize(save_path) in [0, 2]
                except FileNotFoundError:
                    is_empty_file = True

                # conditions for writing a file on the temporary folder:
                # the downloaded data must not be already saved locally.
                if not os.path.exists(save_path) or is_empty_file:
                    # modifying the value of `save_path` to one in a tmp folder
                    save_path = os.path.join(TMP_DIR, filename)
                    with open(save_path, "wt") as file_handler:
                        json.dump(data, file_handler, indent=2)

                artifact.add_file(save_path, filename)

                logger.info(f"Logging `{filename}` artifact.")

                run.log_artifact(artifact)
                artifact.wait()
    else:
        logger.info(f"Saving {filename} at {filepath}")

        with open(save_path, "w") as file:
            json.dump(data, file, indent=2)
