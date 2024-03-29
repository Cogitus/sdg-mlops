import argparse
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from tensorflow import keras
from utils.custom_logging import results_logging
from utils.model import donwload_wandb_datasets, load_datasets, train_model

import wandb

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args: argparse.Namespace) -> None:
    wandb.tensorboard.patch(root_logdir="./tensorboard_logs")

    run = wandb.init(
        job_type="train",
        project="sdg-onu",
        sync_tensorboard=True,
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

    with mlflow.start_run() as mlflow_run:
        constraint = keras.constraints.MaxNorm(max_value=2)
        class_weight_kind = None

        # decrease the learning by a factor of 'rate' every 'decay_steps'
        decay_rate = 1 / args.rate
        lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            args.initial_learning_rate, args.decay_steps, decay_rate
        )

        learning_rate = lr_scheduler

        model_params = {
            "constraint": constraint,
            "class_weight_kind": class_weight_kind,
            "output_sequence_length": args.output_sequence_length,
            "optimizer": args.optimizer,
            "units": args.units,
            "dropout": args.dropout,
            "n_hidden": args.n_hidden,
            "epochs": args.epochs,
            "initial_learning_rate": args.initial_learning_rate,
            "decay_steps": args.decay_steps,
            "rate": args.rate,
            "decay_rate": decay_rate,
        }

        # logging at mlflow
        mlflow.log_params(params=model_params)

        model, history = train_model(
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
            class_weight_kind=class_weight_kind,
            optimizer=args.optimizer,
            learning_rate=learning_rate,
            units=args.units,
            dropout=args.dropout,
            n_hidden=args.n_hidden,
            output_sequence_length=args.output_sequence_length,
            constraint=constraint,
            epochs=args.epochs,
        )

        # logging the metrics of evaluation at wandb and mlflow
        results_logging(
            model=model,
            valid_set=valid_set,
            test_set=test_set,
            wandb_run=run,
            mlflow_log=True,
        )

        # creating the model signature (for mlflow visualization at its UI and
        # at the model register)
        input_schema = Schema(inputs=[TensorSpec(np.dtype(str), (-1,))])
        output_schema = Schema(
            inputs=[
                ColSpec(DataType.string, name=feature_name)
                for feature_name in ["SDG " + str(i + 1) for i in range(16)]
            ]
        )
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # this logs the model to the mlflow tracking server
        mlflow.tensorflow.log_model(
            model,
            "sdg_models",
            signature=signature,
            input_example={
                "text": "Outdoor navigation with four-engined formations using adaptive slider control"
            },
        )

        RUN_ID = mlflow_run.info.run_id
        EXPERIMENT_ID = mlflow.get_experiment_by_name(
            args.experiment_name
        ).experiment_id
        # this expands to something like /tmp/1/lkhasjbdbnas12khad/
        MODEL_FOLDER = Path(PATH_TMP, EXPERIMENT_ID, RUN_ID)

    # this properly saves the model locally for the only purpose of further
    # logginh of it at wandb
    mlflow.tensorflow.save_model(
        model,
        MODEL_FOLDER.as_posix(),
        signature=signature,
        input_example={
            "text": "Outdoor navigation with four-engined formations using adaptive slider control"
        },
    )

    logger.info(f"Saving model of run {RUN_ID}  in W&B located at {MODEL_FOLDER}")

    model_artifact = wandb.Artifact(
        args.model_name, type="model", metadata=model_params
    )
    model_artifact.add_dir(MODEL_FOLDER)
    wandb.run.log_artifact(model_artifact, aliases=["latest"])

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
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)

    ARGS = parser.parse_args()

    os.environ["MLFLOW_EXPERIMENT_NAME"] = ARGS.experiment_name

    main(ARGS)
