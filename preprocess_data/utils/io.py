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
    local_savepath: Path = Path("./artifacts"),
) -> Path:
    """Downloads an artifact from W&B and saves it to a local directory.

    Args:
        artifact_name (str): The name of the artifact to download. This must be
            a W&B-recognizable identifier.
        run (wandb.sdk.wandb_run.Run): The W&B run object for logging purposes.
        local_savepath (pathlib.Path, optional): The local directory to save the artifact.
            Defaults to Path("./artifacts").

    Returns:
        pathlib.Path: The path to the downloaded artifact.
    """
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
    filename: str,
    tags_array: list[str],
    run: Run,
    project_name: str | None = None,
    filepath: Path = Path.cwd(),
    remote: bool = False,
) -> None:
    """
    Save the given data to a file locally or on a W&B project.

    Args:
        data (Any): The data to be saved.
        project_name (str | None, optional): The name of the W&B project to save the
            data to. Defaults to None (if data is to be save remotely).
        filename (str): The name of the file to save the data to.
        tags_array (list[str]): A list of tags to apply to the saved artifact.
        run (wandb.sdk.wandb_run.Run): The W&B run to associate the artifact with.
        filepath (Path, optional): The path to save the file to locally. Defaults to
            the current working directory.
        remote (bool, optional): Whether to save the file on a W&B project. Defaults
            to False.
    """
    # local where the file already is or where is to save it (locally)
    save_path = filepath / filename

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
