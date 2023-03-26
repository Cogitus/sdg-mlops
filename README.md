# 1) Preparing the environment for the first run
Before running, you must have conda installed on your computer. After that, type the following commands to create and activate the virtual environment that will run the pipeline

```bash
$ conda env create -f main_environment.yml
$ conda activate sdg_main_env
```
or simply:

```bash
$ conda env create -f main_environment.yml && conda activate sdg_main_env
```

Finally, log in with ```WandB``` to make the data versioning works and the metrics start to be captured.

```
$ wandb login
```

# 2) Running the pipeline
The full pipeline is run simply by typing the following command on the CLI at the root folder of this project:

```bash
$ mlflow run .
```
Nonetheless, you can change the [default settings](config.yaml) of the project thanks to the flexibility of [*Hydra's package*](https://hydra.cc/docs/intro/), that allows one to override [default's configurations](config.yaml) on each separate run call. If that's the case, then, the paramater ```overriding_configs``` will be receaving a string with the pair ```"configuration=value"``` to change. For example:

```bash
$ mlflow run . -P overriding_configs="project_name=deploy_model"
```

Or in the case of multiple overrides:

```bash
$ mlflow run . -P overriding_configs="project_name=deploy_model model.seed=8795"
```

## Running chosen steps
If you not specify, all the steps of the pipeline will run on the following order:

1) [```download_data```](download_data/run.py)
2) [```transform_data```](transform_data/run.py)
3) [```preprocess_data```](preprocess_data/run.py)
4) [```split_data```](split_data/run.py)
5) [```train```](train/run.py)
6) [```evaluate_model```](evaluate_model/run.py)

But if you want, you can choose to run the steps that you want, although they must follow priority the order above. Thus, if you choose to run ```download_data``` and ```train```, the first experiment (```download_data```) will be executed and then the second (```train```).

The syntax at the CLI that you must use to do so is:

```bash
mlflow run . -P overriding_configs="main.steps2execute='download_data,train'"
```

__OBS__: Note that on that last example, ```model``` is like a dictionary with more than one key inside of it accessible with the ```.``` operator.

__OBS²__: There is a difference between the ```transform_data``` and ```preprocess_data``` steps. Since the main objective of the model is to be applied on academic works written in *portuguese*, one needs to first translate the texts from *portuguese* to *english* before working on the Natural Language Processing steps. Thus, ```transform_data``` is responsible for this and ```preprocess_data``` for the other said steps.

## Another default configurations and configurations override.
The configurations that are being passed to *Hydra* manipulation can be found at the file [```config.yaml```](config.yaml) and since *Hydra* permits it, they can also be overwriten at the CLI running calls.

# Removing the MLflow environments created
To remove the malfunctioning *mlflow* enviroments or to simply reset the local machine to a fresh start condition where you can properly run the project, simply execute the [```reset_envs.sh```](reset_envs.sh) bash script. For this, on your terminal execute the following commands:

```bash
$ chmod u+x reset_envs.sh
```

which turns the script executable, followed by

```bash
$ ./reset_envs.sh
```
that executes the script and removes all *mlflow conda envs*.

## Removing any traces of the project
For this, the past commands to remove the *mlflow* enviroments will be used in adition to the removal of the ```sdg_main_env```. For this, do:

```bash
$ chmod u+x reset_envs.sh
$ ./reset_envs.sh
$ conda remove --name sdg_main_env --all
$ cd .. && rm -rf SDG_classification/
```
__OBS__: Be careful, the last two commands also deletes this repository.