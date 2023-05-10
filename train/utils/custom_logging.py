import os
from pathlib import Path
from time import strftime

import mlflow
import numpy as np
import tensorflow as tf
import wandb
from utils.metrics import (
    exact_match_ratio,
    f1_overall,
    hamming_loss,
    hamming_score,
    precision_overall,
    recall_overall,
)

# for the typehint of the runs
from wandb.sdk.wandb_run import Run


def get_run_logdir(root_logdir: str) -> str:
    """Return a directory path for storing the infos/metrics/logs of TensorBoard.

    Args:
        root_logdir (str): The root directory for storing logs and artifacts.

    Returns:
        str: The path to a directory for storing the logs and artifacts of a single run.
    """
    RUN_ID = strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, RUN_ID)


def results_logging(
    model: tf.keras.Model,
    valid_set: tf.data.Dataset,
    test_set: tf.data.Dataset,
    model_dir: Path | None = None,
    wandb_run: Run | None = None,
    mlflow_log: bool = False,
) -> None:
    """
    This function logs evaluation results and other metrics using the Weights & Biases
    (W&B) platform. It evaluates the model on the validation and test sets, saves the
    model to a specified directory, and logs various metrics to W&B.

    Args:
        model (tf.keras.Model): The trained Keras model to evaluate.
        valid_set (tf.data.Dataset): The validation dataset to evaluate on.
        test_set (tf.data.Dataset): The test dataset to evaluate on.
        model_dir (Path | None): The directory to save the trained model in. Defaults
            to None.
    """
    # Evaluation
    bce, accuracy = model.evaluate(test_set)
    valid_bce, valid_accuracy = model.evaluate(valid_set)

    # save model
    if model_dir is not None:
        model.save(model_dir)

    # Logging metrics
    y_pred = (model.predict(test_set) > 0.5) + 0
    for i, (X, y) in enumerate(test_set):
        if i > 0:
            y_true = np.concatenate((y_true, y.numpy()))
        else:
            y_true = y.numpy()

    em_ratio = exact_match_ratio(y_true, y_pred)
    overall_accuracy = hamming_score(y_true, y_pred)
    overall_loss = hamming_loss(y_true, y_pred)
    precision = precision_overall(y_true, y_pred)
    recall = recall_overall(y_true, y_pred)
    f1 = f1_overall(y_true, y_pred)

    metrics = {
        "Accuracy": accuracy,
        "Validation Accuracy": valid_accuracy,
        "Loss": bce,
        "Validation Loss": valid_bce,
        "Exact Match Ratio": em_ratio,
        "Hamming Score": overall_accuracy,
        "Hamming Loss": overall_loss,
        "Overall Precision": precision,
        "Overall Recall": recall,
        "Overall F1": f1,
    }
    wandb_run.log(metrics)

    # only logs at mlflow if explicitly said
    if mlflow_log:
        mlflow.log_metrics(metrics)
        mlflow.log_param(key="Wandb URL", value=wandb_run.get_url())
