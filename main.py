import logging
import os

import hydra
import mlflow
from dotenv import load_dotenv
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# getting environment variables from .env (the AWS secrets)
load_dotenv()


# this hydra decorator passes the dict config as a DictConfig parameter.
@hydra.main(config_path="conf", config_name="run_configurations", version_base=None)
def run(configuration: DictConfig) -> None:
    # defining the tracking server location
    TRACKING_URI = None
    if configuration["tracking_server"]["default"] == "aws":
        TRACKING_URI = configuration["tracking_server"]["uri"]["aws"]
    else:
        TRACKING_URI = configuration["tracking_server"]["uri"]["local"]
    mlflow.set_tracking_uri(TRACKING_URI)

    # previously setting the wandb configurations
    os.environ["WANDB_PROJECT"] = configuration["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = configuration["main"]["experiment_group"]

    logger.info(
        f'`{configuration["main"]["project_name"]}` set as '
        + "the environment variable WANDB_PROJECT"
    )
    logger.info(
        f'`{configuration["main"]["experiment_group"]}` set as '
        + "the environment variable WANDB_RUN_GROUP"
    )

    # verifying if is the arguments were passed as string or as a list.
    if isinstance(configuration["main"]["steps2execute"], str):
        # this was a comma-separeted string (each term was a step of the list)
        STEPS = configuration["main"]["steps2execute"].split(",")
    else:
        # converting the incoming omegaconf.listconfig.ListConfig to list
        STEPS = list(configuration["main"]["steps2execute"])

    logger.warning(
        "The following steps are going to be executed: "
        + " ".join(["`" + step + "`" for step in STEPS])
    )

    # full path of the project root
    ROOT_PATH = hydra.utils.get_original_cwd()

    # OBS: from here and on, all the steps are listed. Its up to the arguments
    # passed to decide which ones will run.
    if "download_data" in STEPS:
        mlflow.set_experiment("download_data")
        mlflow.projects.run(
            uri=os.path.join(ROOT_PATH, "download_data"),
            entry_point="main",
        )

    if "preprocess_data" in STEPS:
        mlflow.set_experiment("preprocess_data")
        mlflow.projects.run(
            uri=os.path.join(ROOT_PATH, "preprocess_data"),
            entry_point="main",
            parameters={
                "titles": configuration["wandb"]["tag"]["titles"],
                "authors": configuration["wandb"]["tag"]["authors"],
                "affiliations": configuration["wandb"]["tag"]["affiliations"],
                "dois": configuration["wandb"]["tag"]["dois"],
                "keywords": configuration["wandb"]["tag"]["keywords"],
                "abstracts": configuration["wandb"]["tag"]["abstracts"],
            },
        )

    if "split_data" in STEPS:
        PIPELINE_PROGRAM = [
            {"entry_point": "download_language_models", "parameters": {}},
            {
                "entry_point": "dataset_balancing",
                "parameters": {
                    "quantile": configuration["split_data"]["balancing"]["quantile"],
                    "random_state": configuration["split_data"]["balancing"][
                        "random_state"
                    ],
                    "path_sdg_dataset": configuration["split_data"]["splitting"][
                        "path_sdg_dataset"
                    ],
                },
            },
            {
                "entry_point": "preprocess_and_split",
                "parameters": {
                    "train_size_factor": configuration["split_data"]["splitting"][
                        "train_size_factor"
                    ],
                    "dataset_name": configuration["wandb"]["tag"]["balanced_dataset"],
                    "test_share_size": configuration["split_data"]["splitting"][
                        "test_share_size"
                    ],
                    "random_state": configuration["split_data"]["splitting"][
                        "random_state"
                    ],
                },
            },
            {
                "entry_point": "advanced_preprocessing",
                "parameters": {
                    "X_train": configuration["wandb"]["tag"]["X_train"],
                    "y_train": configuration["wandb"]["tag"]["y_train"],
                    "X_valid": configuration["wandb"]["tag"]["X_valid"],
                    "y_valid": configuration["wandb"]["tag"]["y_valid"],
                    "X_test": configuration["wandb"]["tag"]["X_test"],
                    "y_test": configuration["wandb"]["tag"]["y_test"],
                    "tf_batch_size": configuration["split_data"][
                        "advanced_preprocessing"
                    ]["tf_batch_size"],
                    "tf_seed": configuration["split_data"]["advanced_preprocessing"][
                        "tf_seed"
                    ],
                },
            },
        ]

        mlflow.set_experiment("split_data")
        for execution_step in PIPELINE_PROGRAM:
            mlflow.projects.run(
                uri=os.path.join(ROOT_PATH, "split_data"),
                entry_point=execution_step["entry_point"],
                parameters=execution_step["parameters"],
            )

    if "train" in STEPS:
        mlflow.set_experiment("train")
        mlflow.projects.run(
            uri=os.path.join(ROOT_PATH, "train"),
            entry_point="main",
            experiment_name="train",
            parameters={
                "tensorflow_datasets": configuration["train"]["tensorflow_datasets"],
                "output_sequence_length": configuration["train"][
                    "output_sequence_length"
                ],
                "optimizer": configuration["train"]["optimizer"],
                "units": configuration["train"]["units"],
                "dropout": configuration["train"]["dropout"],
                "n_hidden": configuration["train"]["n_hidden"],
                "epochs": configuration["train"]["epochs"],
                "initial_learning_rate": configuration["train"][
                    "initial_learning_rate"
                ],
                "decay_steps": configuration["train"]["decay_steps"],
                "rate": configuration["train"]["rate"],
                "model_name": configuration["train"]["model_name"],
            },
        )

    if "evaluate_model" in STEPS:
        PIPELINE_PROGRAM = [
            {"entry_point": "download_language_models", "parameters": {}},
            {
                "entry_point": "main",
                "parameters": {
                    "model_tag": configuration["evaluate_model"]["model_tag"],
                    "input_data_tag": configuration["evaluate_model"]["input_data_tag"],
                },
            },
        ]

        mlflow.set_experiment("evaluate_model")
        for execution_step in PIPELINE_PROGRAM:
            mlflow.projects.run(
                uri=os.path.join(ROOT_PATH, "evaluate_model"),
                entry_point=execution_step["entry_point"],
                parameters=execution_step["parameters"],
            )


if __name__ == "__main__":
    run()
