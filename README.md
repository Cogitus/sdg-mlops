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

## Preparing the database for storing the logs
Originally, `mlflow` stores the runs' info at `./mlruns/` folder. There, each folder, e.g., `./mlruns/0/` (that is the "Default" one) represents an [`experiment`](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments). The purpose of the following configuration is to make all the data that is not related to the trained models to be logged at a `postgresql` database. If you want to decouple it even more, you can store the artifacts (models) at a separated database (like an AWS S3 bucket). Anyway, the configurations is as it follows:

install postegresql
```bash
$ sudo apt update
# download .pgp key, convert to .gpg, save it in /etc/apt/keyrings
$ curl -L https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor | sudo tee /etc/apt/keyrings/postgresql-archive-keyring.gpg >/dev/null
# add correspondent .list file to /etc/apt/sources.list.d/
$ echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/postgresql-archive-keyring.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" | sudo tee  /etc/apt/sources.list.d/postgresql.list
$ sudo apt update
$ sudo apt install postgresql # this downloads the latest version
# enable it to initialize with Linux
$ sudo systemctl start postgresql
```

access postgres
```bash
$ sudo -i -u postgres
$ psql
```


While inside postgres
```sql
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
```

finally, create the directory for the the Tracking Server to log the machine learning models and other artifacts
```bash
# execute this command at the root of this repository
$ mkdir ./mlruns
```

# 2) Running the pipeline
Initialize the mlflow server (this step simply indicates to all calls to mlflow runs where the artifacts/models will be save and where the metrics will be logged):
```bash
$ mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root file:<PATH_TO_THE_REPO>/SDG_classification/mlruns -h 0.0.0.0 -p 8000
```

![mlflow server appearence](./imgs/mlflow_interface.png)

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

![run info appearence](./imgs/mlflow_run_info.png)

![wandb interface](./imgs/wandb_interface.png)

## But what if you want to run a specific entrypoint?
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

# IMPORTANT CONSIDERATION
You don't need to run `mlflow server` in order to execute the `mlflow run` calls. But doing that way you will have everything being save at `./mlruns/` folder. And if you really want to do it, comment the line at ./main.py that says:

```python
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:8000"
```

# References
Dias, A., Alves, G., Lima, G., Alsina, A. and Silva, I. (n.d.). Abordagem Cientométrica Orientada a Dados para Classificação Multi-Alvo dos Objetivos de Desenvolvimento Sustentável na Automação. [online] Available at: https://www.sba.org.br/cba2022/wp-content/uploads/artigos_cba2022/paper_4112.pdf [Accessed 25 Apr. 2023].
