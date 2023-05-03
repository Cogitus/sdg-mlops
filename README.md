# What is the purpose of this?
This repository presents an implementation of a multilabel classification solution for scientometric data that classifies inputs into one or more Sustainable Development Goals (SDGs) from the United Nations General Assembly. The model was developed by Alexandre Dias who deserve all the credits for this work.

The goal of this repo, on other way, is to apply the model already developed by the author to a MLops paradigm, making it easier to deploy and maintain.

The Sustainable Development Goals are a set of 17 interconnected goals adopted by the United Nations General Assembly to end poverty, protect the planet, and ensure that all people enjoy peace and prosperity by 2030, although in this solution only 16 will taken into consideration. The accurate classification of scientometric data into the SDGs can contribute to the advancement of these goals.

For further details on the methodology and results of the model, please refer to the original work [here](https://www.sba.org.br/cba2022/wp-content/uploads/artigos_cba2022/paper_4112.pdf).

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
$ mlflow run . -P overriding_configs="train.n_hidden=2"
```

Or in the case of multiple overrides:

```bash
$ mlflow run . -P overriding_configs="train.units=50 train.epochs=6 train.rate=3 train.n_hidden=2"
```

## But what i you want to run a specific entrypoint?
For this, the step can be specified as another parameter to mlflow. In that case, the parameter `step`. For this, is good to remeber that in need to be preceeded by the `-P` at the CLI call. For example:

```bash
$ mlflow run . -P steps='train'
# or with hydra overridings:
$ mlflow run . -P steps='train' -P overriding_configs="train.units=50 train.epochs=6 train.rate=3 train.n_hidden=2"
```

NOTE: For default, the only entrypoints that will execute at the "full" pipeline are `train` and `evaluate_model`. If one wants to execute more than this, then, it will be needed to be explicitly specified, as in:

```bash
$ mlflow run . -P steps='download_data,split_data,train'
```

## Another default configurations and configurations override.
The configurations that are being passed to *Hydra* manipulation can be found at the file [```config.yaml```](config.yaml) and since *Hydra* permits it, they can also be overwriten at the CLI running calls. If you want to know more about the possibilities of overriding the current configuration, check the [hydra's documentation about this](https://hydra.cc/docs/advanced/override_grammar/basic/).

## Multirun in hydra
Since hydra has the multirun feature, it can be easily done by specifying the `-m` parameter at the `overriding_configs` parameter. In an easier way:

```bash
$ mlflow run . -P steps='train' -P overriding_configs="train.epochs=6,8 train.rate=3,4,5 train.n_hidden=2 -m"
```

Note that `train.rate` and `train.epochs` have more than one value, so, when using `-m` hydra will execute multiple runs.

## Adding new arguments to `/config/run_configurations.yaml`
If one wants to change `run_configurations.yaml` by removing or adding an argument for a given entrypoint, he'll need to do a serious of recursive changes:

By *adding* a parameter:
1) Add the parameter to `/config/run_configurations.yaml`
2) Add the parameter to `parameter` dict() of the `mlflow.projects.run()` call for the desired entrypoint
3) Add the parameter to the `parameters` section of entrypoint's `MLproject` file.
4) Add the parameter as an `argparser` argument with `parser.add_argument()` at the file that is called at the entrypoint python call in the section `command`

The same thing is done by *removing* a parameter, except that, obviously, you'll remove instead of adding.

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

# References
Dias, A., Alves, G., Lima, G., Alsina, A. and Silva, I. (n.d.). Abordagem Cientométrica Orientada a Dados para Classificação Multi-Alvo dos Objetivos de Desenvolvimento Sustentável na Automação. [online] Available at: https://www.sba.org.br/cba2022/wp-content/uploads/artigos_cba2022/paper_4112.pdf [Accessed 25 Apr. 2023].
