from multiprocessing import Pool, cpu_count

import numpy as np
import tensorflow as tf


def process_data(data: tf.Tensor) -> np.ndarray:
    return data.numpy()


def convert_dataset_to_numpy(
    dataset: tf.data.Dataset, column_type: str = "inputs"
) -> np.ndarray:
    """Converts part of the structure of a Tensorflow dataset, which are iterables
    of tf.Tensor to a np.ndarray. The conversion is parallelized using multiple
    CPUs whereas the number is (maximum_number_cpus - 1)

    Args:
        dataset (tf.data.Dataset): Tensorflow dataset that will be processed
        column_type (str, optional): Selects if one wants to convert the input
            data to np.darray or the labels/classes to it. Defaults to 'inputs'.

    Returns:
        np.ndarray: Converted part of the dataset as a np array. Note that if the `column_type`
            is 'labels', this will be a np.ndarray[np.ndarray[float]], otherwise
            a np.ndarray[str]
    """
    assert column_type in [
        "inputs",
        "labels",
    ], "column_type must be 'inputs' or 'labels"
    assert isinstance(dataset, tf.data.Dataset)

    NUM_CPUS = cpu_count() - 1

    pool = Pool(NUM_CPUS)
    # processes the data
    if column_type == "inputs":
        results = pool.map(process_data, [data for data, _ in dataset])
    # processes the labels
    else:
        results = pool.map(process_data, [labels for _, labels in dataset])
    pool.close()

    return np.concatenate(results)


def build_text_vectorization_layer(
    _train_set: tf.data.Dataset,
    output_sequence_length: int,
    max_vocabulary_size: int = 20000,
) -> tf.keras.layers.TextVectorization:
    """
    This function builds a Keras TextVectorization layer based on the training set.

    Args:
        _train_set (tf.data.Dataset): A TensorFlow dataset object representing the training set.
        output_sequence_length (int): The output sequence length of the TextVectorization layer.
        max_vocabulary_size (int, optional): The maximum vocabulary size. Defaults to 20000.

    Returns:
        tf.keras.layers.TextVectorization: A Keras TextVectorization layer after `.adapt()`.
    """
    X_train = convert_dataset_to_numpy(_train_set)

    text_vectorization = tf.keras.layers.TextVectorization(
        max_tokens=max_vocabulary_size,
        output_mode="int",
        output_sequence_length=output_sequence_length,
    )

    # "adapt" the layer to the data (this is the same as "fit")
    text_vectorization.adapt(X_train)

    return text_vectorization
