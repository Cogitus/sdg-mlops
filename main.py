import os
import logging

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


# this hydra decorator passes the dict config as a DictConfig parameter.
@hydra.main(config_name='run_configurations')
def run(configuration:DictConfig) -> None:
    # previously setting the wandb configurations
    os.environ["WANDB_PROJECT"] = configuration["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = configuration["main"]["experiment_name"]

    logger.info(f'`{configuration["main"]["project_name"]}` set as ' +
                'the environment variable WANDB_PROJECT')
    logger.info(f'`{configuration["main"]["experiment_name"]}` set as ' +
                'the environment variable WANDB_RUN_GROUP')


    # full path of the project root
    ROOT_PATH = hydra.utils.get_original_cwd()


    # verifying if is the arguments were passed as string or as a list.
    if isinstance(configuration['main']['steps2execute'], str):
        # this was a comma-separeted string (each term was a step of the list)
        STEPS = configuration['main']['steps2execute'].split(',')
    else:
        # converting the incoming omegaconf.listconfig.ListConfig to list
        STEPS = list(configuration["main"]["steps2execute"])

    logger.warning('The following steps are going to be executed: ' +
                    ' '.join(['`' + step + '`' for step in STEPS]))

    # OBS: from here and on, all the steps are listed. Its up to the arguments 
    # passed to decide which ones will run.
    
    if 'download_data' in STEPS:
        mlflow.run(
            os.path.join(ROOT_PATH, 'download_data'),
            entry_point='main',
            parameters={
                'parametro_teste': 'teste'
            }
        )

    # if 'preprocess_data' in STEPS:
    #     mlflow.run(
    #         os.path.join(ROOT_PATH, 'preprocess_data'),
    #         entry_point=configuration["main"]['environment_type'],
    #         parameters={

    #         }
    #     )

    # if 'split_data' in STEPS:
    #     mlflow.run(
    #         os.path.join(ROOT_PATH, 'split_data'),
    #         entry_point=configuration["main"]['environment_type'],
    #         parameters={

    #         }
    #     )

    # if 'train' in STEPS:
    #     mlflow.run(
    #         os.path.join(ROOT_PATH, 'train'),
    #         entry_point=configuration["main"]['environment_type'],
    #         parameters={

    #         }
    #     )

    # if 'evaluate_model' in STEPS:
    #     mlflow.run(
    #         os.path.join(ROOT_PATH, 'evaluate_model'),
    #         entry_point=configuration["main"]['environment_type'],
    #         parameters={

    #         }
    #     )


if __name__ == '__main__':
    run()