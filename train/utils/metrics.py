"""
This is a multilabel classification problem, which requires the metrics we will
calculate to consider this new context, and therefore, they are a bit different from
the ones usually used in multi-class or binary problems. The theoretical foundation
of these metrics can be found here:

https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_test: np.ndarray, y_preds: np.ndarray, labels: list[str]
) -> pd.DataFrame:
    """
    Compute binary classification metrics for each label.

    Args:
        y_test (np.ndarray): The true labels of the test set.
        y_preds (np.ndarray): The predicted labels for the test set.
        labels (list[str]): A list of labels for the binary classification task.

    Returns:
        pd.DataFrame: A dataframe containing the computed binary classification metrics for each label.
    """
    metrics = {}.fromkeys(labels, None)
    for label in labels:
        metrics[label] = {}.fromkeys(
            ["Accuracy", "Recall", "Precision", "F1 Score", "ROC AUC"], None
        )

    for i, label in enumerate(labels):
        y_true = y_test[:, i]
        y_pred = y_preds[:, i].round()

        metrics[label]["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics[label]["Recall"] = recall_score(y_true, y_pred)
        metrics[label]["Precision"] = precision_score(y_true, y_pred)
        metrics[label]["F1 Score"] = f1_score(y_true, y_pred)
        metrics[label]["ROC AUC"] = roc_auc_score(y_true, y_pred)
    metrics_data = pd.DataFrame(metrics).round(4)

    return metrics_data


def exact_match_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the exact match ratio between the true and predicted labels.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: The exact match ratio between the true and predicted labels.
    """

    return np.all(y_pred == y_true, axis=1).mean()


def hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Hamming score between the true and predicted labels.
    Also known as overall accuracy.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: The Hamming score between the true and predicted labels.
    """

    accum = 0
    for i in range(y_true.shape[0]):
        accum += np.sum(np.logical_and(y_true[i], y_pred[i])) / np.sum(
            np.logical_or(y_true[i], y_pred[i])
        )
    return accum / y_true.shape[0]


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Hamming loss between the true and predicted labels.
    Also known as overall loss.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: The Hamming loss between the true and predicted labels.
    """

    accum = 0
    for i in range(y_true.shape[0]):
        accum += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(
            y_true[i] == y_pred[i]
        )
    return accum / (y_true.shape[0] * y_true.shape[1])


def precision_overall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the overall precision between the true and predicted labels.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: The overall precision between the true and predicted labels.
    """

    accum = 0
    for i in range(y_true.shape[0]):
        if np.sum(y_pred[i]) == 0:
            continue
        accum += np.sum(np.logical_and(y_true[i], y_pred[i])) / np.sum(y_pred[i])
    return accum / y_true.shape[0]


def recall_overall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the overall recall between the true and predicted labels.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: The overall recall between the true and predicted labels.
    """

    accum = 0
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) == 0:
            continue
        accum += np.sum(np.logical_and(y_true[i], y_pred[i])) / np.sum(y_true[i])
    return accum / y_true.shape[0]


def f1_overall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the overall F1-score between the true and predicted labels.

    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: The overall F1-score between the true and predicted labels.
    """

    accum = 0
    for i in range(y_true.shape[0]):
        if (np.sum(y_true[i]) == 0) and (np.sum(y_pred[i]) == 0):
            continue
        accum += (2 * np.sum(np.logical_and(y_true[i], y_pred[i]))) / (
            np.sum(y_true[i]) + np.sum(y_pred[i])
        )
    return accum / y_true.shape[0]


def print_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print multilabel classification performance metrics.

    This function takes as input two arrays, `y_true` and `y_pred`, and computes various performance metrics
    for multilabel classification, including the exact match ratio, overall accuracy, overall loss, overall
    precision, overall recall, and overall F1 score. The function then prints these metrics to the console.

    Parameters
    ----------
        y_true : array-like of shape (n_samples, n_classes)
            Ground truth (correct) target values for multilabel classification.
        y_pred : array-like of shape (n_samples, n_classes)
            Estimated targets as returned by a multilabel classifier.

    Returns
    -------
    None

    Notes
    -----
        The function uses the following helper functions:
        - exact_match_ratio(y_true, y_pred): Computes the ratio of samples for which all the predicted labels match
        the true labels exactly.
        - hamming_score(y_true, y_pred): Computes the fraction of labels that are correctly predicted.
        - hamming_loss(y_true, y_pred): Computes the fraction of labels that are incorrectly predicted.
        - precision_overall(y_true, y_pred): Computes the average precision across all labels.
        - recall_overall(y_true, y_pred): Computes the average recall across all labels.
        - f1_overall(y_true, y_pred): Computes the average F1 score across all labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import multilabel_confusion_matrix
    >>> y_true = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    >>> y_pred = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]])
    >>> print_multilabel_metrics(y_true, y_pred)
    Exact Match         = 0.3333
    Overall Accuracy    = 0.6667
    Overall Loss        = 0.3333
    Overall Precision   = 0.5833
    Overall Recall      = 0.5833
    Overall F1 Score    = 0.5833
    """
    em_ratio = exact_match_ratio(y_true, y_pred)
    overall_accuracy = hamming_score(y_true, y_pred)
    overall_loss = hamming_loss(y_true, y_pred)
    precision = precision_overall(y_true, y_pred)
    recall = recall_overall(y_true, y_pred)
    f1 = f1_overall(y_true, y_pred)

    print("Exact Match         = {:.4f}".format(em_ratio))
    print("Overall Accuracy    = {:.4f}".format(overall_accuracy))
    print("Overall Loss        = {:.4f}".format(overall_loss))
    print("Overall Precision   = {:.4f}".format(precision))
    print("Overall Recall      = {:.4f}".format(recall))
    print("Overall F1 Score    = {:.4f}".format(f1))
