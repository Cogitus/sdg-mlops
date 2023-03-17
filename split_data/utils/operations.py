import logging
from pathlib import Path

import numpy as np
import pandas as pd

# for the typehint of the runs
from wandb.sdk.wandb_run import Run

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def balance_multilabel_dataset(
    dataset: pd.DataFrame,
    quantile: float = 0.5,
    random_state: int = 42,
    label_columns: list = None,
) -> pd.DataFrame:
    """
    Balance the counts of target labels in a multilabel dataset.

    The function balances the counts of target labels in a dataset by iteratively sampling instances from
    classes with more instances until all classes have the same count of instances. The sample is chosen so
    that it maintains the ratio of labels within the target columns.

    Parameters:
    dataset (dataframe): The unbalanced dataset
    quantile (float, optional): the quantile of the counts of labels at which to balance the dataset.
    If you choose 1, the dataset will remain the same. If you choose 0, than the dataset will be almost
    as balanced as possible. however, you'll probably lose many data. Default value is 0.5 (median).
    random_state (int, optional): the random seed used to sample the dataframe. Default is 42.

    Returns:
    dataset (dataframe): a balanced dataset.
    Example:
    >>> unbalanced_dataset = pd.read_csv("unbalanced_dataset.csv")
    >>> balanced_dataset = balance_multilabel_dataset(unbalanced_dataset)
    """
    if label_columns is None:
        raise ValueError("The dataset columns must be specified")
    logger.info(
        f"starting balancing for quantile={quantile} and random_state={random_state}"
    )
    # compute the overall counts of labels in the dataset before multilabel balancing
    sdg_counts = dataset.loc[:, label_columns].sum(axis="index")

    # compute the quantile label count and identify those labels below it
    quantile_label_count = np.quantile(sdg_counts, q=quantile)

    # compute the number of samples to add from each label to reach the quantile of the
    # number of samples
    samples_to_add = quantile_label_count - sdg_counts
    samples_to_add[samples_to_add < 0] = quantile_label_count

    # sort label from the minimum to maximum label count
    sorted_labels = sdg_counts.sort_values().index.tolist()

    balanced_dataset = pd.DataFrame(columns=dataset.columns)

    for label in sorted_labels:
        # compute the number of records that must be sampled for current label
        label_samples_to_add = int(samples_to_add[label])
        samples_available = np.sum(dataset[label] == 1)

        label_has_samples_available = samples_available > 0
        label_needs_more_samples = label_samples_to_add > 0

        if label_has_samples_available and label_needs_more_samples:
            # creates a mask to filter the dataset with the samples from current label
            samples_from_selected_label = dataset[label] == 1

            # guarantee that it will not try to add more samples than there is available
            if label_samples_to_add > samples_available:
                label_samples_to_add = samples_available

            # samples the dataset
            selected_samples = dataset[samples_from_selected_label].sample(
                n=label_samples_to_add, random_state=random_state
            )

            # remove the selected samples from the dataset in order
            # to avoid those samples in the next iteration
            dataset = dataset[~samples_from_selected_label]

            # concatenate the balanced_dataset with the selected_samples
            balanced_dataset = pd.concat([balanced_dataset, selected_samples])

        # update the counts of samples to be added to the next labels
        balanced_label_count = balanced_dataset.loc[:, label_columns].sum(axis="index")
        samples_to_add = quantile_label_count - balanced_label_count

    return balanced_dataset


def download_wandb_data(
    artifact_name: str,
    run: Run,
    local_savepath: Path = Path("./artifacts"),
    artifact_type: str = "dataset",
) -> pd.DataFrame:
    logger.info(
        f"loading the `{artifact_name}` from W&B to {local_savepath.absolute()}"
    )

    # import artifact from wandb
    artifact = run.use_artifact(artifact_name, type=artifact_type)
    local_savepath = artifact.download(local_savepath)

    # get the donwloaded file name by its name at W&B (such as y_train:v1)
    artifact_name = lambda x: x.split(":")[0] + ".csv"

    return pd.read_csv(Path(local_savepath) / artifact_name(artifact.name), index_col=0)
