import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tensorflow import keras
from wandb.sdk.wandb_run import Run

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_artifact(
    wandb_tag: str,
    wandb_run: Run,
    local_savepath: Path = Path("./artifacts"),
    artifact_type: str = "model",
) -> Path:
    """Downloads a specific artifact from Weights & Biases and returns its content.

    Args:
        wandb_tag (str): The name of the artifact to download from Weights & Biases.
            Note that this must be a W&B-style recognizable string.
        wandb_run (Run):The wandb Run object where the artifact is stored.
        local_savepath (Path, optional): The local directory to download the
            artifact to. Defaults to Path("./artifacts").
        artifact_type (str, optional): The type of artifact to download.
            Defaults to "model".

    Returns:
        Path: Path at the local filesystem to the downloaded artifact.
    """
    logger.info(f"loading the `{wandb_tag}` from W&B to {local_savepath.absolute()}")

    # import artifact from wandb
    artifact = wandb_run.use_artifact(wandb_tag, type=artifact_type)
    local_savepath = artifact.download(local_savepath)

    return local_savepath
    ...


def get_weighted_loss(
    weights: np.ndarray[np.ndarray[float]],
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Weighted binary cross entropy loss function se custom loss function

    >>> to see more:
    https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras

    Args:
        weights (np.ndarray[np.ndarray[float]]): Values that will weight the label
            at the custom loss calculation

    Returns:
        Callable[[np.ndarray, np.ndarray], float]: Custom loss function that is created
    """

    def weighted_loss(y_train: np.ndarray, y_pred: np.ndarray):
        return keras.backend.mean(
            (weights[:, 0] ** (1 - y_train))
            * (weights[:, 1] ** (y_train))
            * keras.backend.binary_crossentropy(y_train, y_pred),
            axis=-1,
        )

    return weighted_loss
