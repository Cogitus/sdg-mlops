from collections.abc import Callable
from multiprocessing import Pool, cpu_count

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from vectorization import convert_dataset_to_numpy


def compute_class_weights(train_set: tf.data.Dataset) -> np.ndarray[np.ndarray[float]]:
    """This function assigns weights to each class when calculating the binary_crossentropy
    loss function.
    The weights of each class can be automatically calculated with the compute_class_weights
    function, or we can simply pass the weights manually to the get_weighted_loss function,
    which then returns the binary_crossentropy function considering the weights that
    were passed as parameters.

    Although there is already a stage of for dataset balancing, a cost function
    can still be adapted to consider slight imbalances between classes.

    Args:
        train_set (tf.data.Dataset): tensorflow dataset which will be computed the
            labels/class weights since its natural unbalancing

    Returns:
        np.ndarray: the class weights for balancing the input dataset
    """
    y_train = convert_dataset_to_numpy(train_set, column_type="labels")

    N_CLASSES = y_train.shape[1]

    weights = np.empty([N_CLASSES, 2])
    for i in range(N_CLASSES):
        weights[i] = compute_class_weight(
            "balanced", classes=[0.0, 1.0], y=y_train[:, i]
        )
    return weights


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
            considering the `weights` input
    """

    def weighted_loss(y_train: np.ndarray, y_pred: np.ndarray):
        return keras.backend.mean(
            (weights[:, 0] ** (1 - y_train))
            * (weights[:, 1] ** (y_train))
            * keras.backend.binary_crossentropy(y_train, y_pred),
            axis=-1,
        )

    return weighted_loss


def get_class_weight(
    train_set: tf.data.Dataset, class_weight_kind: str | None = "balanced"
) -> np.ndarray:
    """Caculates the class weights for a dataset. There are 3 default strategies available
    for weighting classes and dealing with imbalances between them, which are:
        1) Use a standard cost function and equally weigh all classes;
        2) Use a custom cost function with weights calculated automatically;
        3) Use a custom cost function with weights calculated manually.

    Args:
        train_set (tf.data.Dataset): Dataset which the class weights will be extracted.
        class_weight_kind (str | None): The class weight type of calculation.
            Defaults to "balanced".

    Returns:
        np.ndarray: Calculated class weights for the input dataset.
    """
    if class_weight_kind is None:
        class_weights = None
    elif class_weight_kind == "balanced":
        class_weights = compute_class_weights(train_set)
    elif class_weight_kind == "two-to-one":
        class_weights = np.zeros((16, 2))

        class_weights[:, 0] = 1.0
        class_weights[:, 1] = 2.0

    return class_weights
