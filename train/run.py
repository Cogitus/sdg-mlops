import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import wandb
from tensorflow import keras
from utils.model import donwload_wandb_datasets, load_datasets, train_model

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args: argparse.Namespace) -> None:
    run = wandb.init(
        job_type="train",
        project="sdg-onu",
        tags=[
            "dev",
            "tf",
            "train",
            "callback",
            "tensorflow",
            "keras",
            "training",
        ],
    )

    tmp_dir = TemporaryDirectory()
    PATH_TMP = Path(tmp_dir.name)

    # downloading datasets
    donwload_wandb_datasets(args.tensorflow_datasets, run=run, local_savepath=PATH_TMP)
    train_set, valid_set, test_set = load_datasets(datapath=PATH_TMP)

    with mlflow.start_run():
        constraint = keras.constraints.MaxNorm(max_value=2)
        class_weight_kind = None
        output_sequence_length = args.output_sequence_length
        optimizer = args.optimizer
        units = args.unit
        dropout = args.dropout
        n_hidden = args.n_hidden
        epochs = args.epochs

        initial_learning_rate = args.initial_learning_rate
        decay_steps = args.decay_steps  # number of steps per epoch
        rate = args.rate
        # decrease the learning by a factor of 'rate' every 'decay_steps'
        decay_rate = 1 / rate
        lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps, decay_rate
        )

        learning_rate = lr_scheduler

        # logging at mlflow
        mlflow.log_params(
            params={
                "constraint": constraint,
                "class_weight_kind": class_weight_kind,
                "output_sequence_length": output_sequence_length,
                "optimizer": optimizer,
                "units": units,
                "dropout": dropout,
                "n_hidden": n_hidden,
                "epochs": epochs,
                "initial_learning_rate": initial_learning_rate,
                "decay_steps": decay_steps,
                "rate": rate,
                "decay_rate": decay_rate,
            }
        )

        model, history = train_model(
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
            class_weight_kind=class_weight_kind,
            optimizer=optimizer,
            learning_rate=learning_rate,
            units=units,
            dropout=dropout,
            n_hidden=n_hidden,
            output_sequence_length=output_sequence_length,
            constraint=constraint,
            epochs=epochs,
        )

        mlflow.tensorflow.log_model(model, "sdg_models")

    tmp_dir.cleanup()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and train a model for multilabel classification"
    )

    parser.add_argument(
        "--tensorflow_datasets",
        type=str,
        help="Path on W&B to the artifact folder that holds test/valid/train tensorflow datasets",
        required=True,
    )
    parser.add_argument("--output_sequence_length", type=int, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--units", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--n_hidden", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--initial_learning_rate", type=float, required=True)
    parser.add_argument("--decay_steps", type=int, required=True)
    parser.add_argument("--rate", type=int, required=True)
    ARGS = parser.parse_args()

    main(ARGS)
