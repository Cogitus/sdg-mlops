import os
from pathlib import Path

import tensorflow as tf
import wandb
from custom_logging import get_run_logdir, results_logging
from tensorflow import keras
from vectorization import build_text_vectorization_layer


# load pre-saved tensorflow datasets
def load_datasets(
    datapath: Path,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Loads pre-saved tensorflow datasets

    Args:
        datapath (Path): Location of the the folder containing the tensorflow
            folders that stores the datasets information and data.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: All the 3 datasets
            for training, train/valid/test
    """
    train_set = tf.data.Dataset.load((datapath / "train_set").as_posix())
    valid_set = tf.data.Dataset.load((datapath / "valid_set").as_posix())
    test_set = tf.data.Dataset.load((datapath / "test_set").as_posix())

    return train_set, valid_set, test_set


def build_model(
    n_outputs: int,
    text_vectorization: tf.keras.layers.TextVectorization,
    optimizer: tf.keras.optimizers.Optimizer | str,
    loss: tf.keras.losses.Loss | str,
    units: int = 50,
    dropout: float = 0.25,
    constraint: tf.keras.constraints.Constraint | None = None,
    n_hidden: int = 2,
) -> tf.keras.Model:
    """
    Builds a deep learning model for text classification using TensorFlow and Keras.

    Args:
        n_outputs (int): The number of output classes.
        text_vectorization (tf.keras.layers.TextVectorization): The text vectorization layer.
        optimizer (tf.keras.optimizers.Optimizer | str): The optimizer to use for training the model.
        loss (tf.keras.losses.Loss | str): The loss function to use for training the model.
        units (int): The number of units per hidden layer. Defaults to 50.
        dropout (float): The dropout rate to use in the hidden layers. Defaults to 0.25.
        constraint (tf.keras.constraints.Constraint | None): An optional constraint
            to apply to the model weights. Defaults to None.
        n_hidden (int): The number of  hidden layers to use. Defaults to 2.

    Returns:
        tf.keras.Model: The deep learning model for text classification.
    """

    # add 1 for the padding token
    vocabulary_size = len(text_vectorization.get_vocabulary()) + 1
    number_out_of_vocabulary_buckets = (
        1  # default value for the text vectorizarization layer, do not change
    )

    # set embedding dimensions to the number of units
    embed_size = units

    # same droupout and constraint rate for the recurrent states
    recurrent_dropout = 0  # must be set to 0 when using GPU
    recurrent_constraint = constraint

    # instantiate the model and add text vectorization and embedding layer
    model = keras.models.Sequential()
    model.add(text_vectorization)
    model.add(
        keras.layers.Embedding(
            input_dim=vocabulary_size + number_out_of_vocabulary_buckets,
            output_dim=embed_size,
            mask_zero=True,
            input_shape=[None],
        )
    )

    # add hidden layers
    for layer in range(n_hidden - 1):
        model.add(
            keras.layers.GRU(
                units,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                kernel_constraint=constraint,
                recurrent_constraint=recurrent_constraint,
            )
        )

    model.add(
        keras.layers.GRU(
            units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_constraint=constraint,
            recurrent_constraint=recurrent_constraint,
        )
    )

    # add output layer
    model.add(keras.layers.Dense(n_outputs, activation="sigmoid"))

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model


def train_model(
    train_set,
    valid_set,
    test_set,
    class_weight_kind,
    optimizer,
    learning_rate,
    units,
    dropout,
    constraint,
    n_hidden,
    output_sequence_length,
    epochs: int = 50,
    log: bool = False,
):
    # train preparation
    text_vectorization_layer = build_text_vectorization_layer(
        train_set, output_sequence_length
    )

    # Loss function
    class_weights = get_class_weight(train_set, class_weight_kind)

    if class_weight_kind is None:
        loss = "binary_crossentropy"
    elif (class_weight_kind == "balanced") or (class_weight_kind == "two-to-one"):
        loss = get_weighted_loss(class_weights)

    # Optimizer
    if optimizer == "Nadam":
        optimizer = keras.optimizers.Nadam(learning_rate, clipnorm=1)
    elif optimizer == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate, clipnorm=1)
    elif optimizer == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate, clipnorm=1, centered=True)

    n_outputs = 16
    model = build_model(
        n_outputs,
        text_vectorization_layer,
        optimizer,
        loss,
        units,
        dropout,
        constraint,
        n_hidden,
    )

    # Define callbacks
    root_logdir = os.path.join(os.curdir, "my_logs")
    run_logdir = get_run_logdir(root_logdir)
    model_dir = os.path.join(run_logdir, "model")

    callbacks = [
        keras.callbacks.TensorBoard(run_logdir),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=2, restore_best_weights=True
        ),
        wandb.keras.WandbCallback(),
    ]

    # Fit model
    history = model.fit(
        train_set, validation_data=valid_set, epochs=epochs, callbacks=callbacks
    )

    # Logging metrics
    if log:
        results_logging(model, valid_set, test_set, model_dir)
    else:
        # Evaluation
        bce, accuracy = model.evaluate(test_set)
        valid_bce, valid_accuracy = model.evaluate(valid_set)

    return model, history
