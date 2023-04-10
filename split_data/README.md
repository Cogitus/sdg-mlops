# Why so many entrypoints?
When looking at the [`MLproject`](MLproject) file, one will repair that there are 4 entrypoints: [`download_language_models`](download_spacy_model.py), [`dataset_balancing`](data_balancing.py), [`preprocess_and_split`](splitting.py) and [`advanced_preprocessing`](advanced_preprocessing.py). They will be run at the same order as previously described and each one of them is responsible for a important step at preparing the following training step (of the full pipeline found at [`main.py`](../main.py))

# Entrypoints
## 1) [`download_language_models`](download_spacy_model.py)
This step is necessary to download at hardcode level the `spacy`'s available trained pipelines for English `en_core_web_lg`. It seems a bit clumsy to put the download of a simples file that could be done with `$pip` inside a `mlflow` entrypoint, but this is a workaround to the impossibility of `conda` to download and to install it at its virtual environment for a mlflow run (e.g, and env with a name such as `mlflow-acf304fd3c8f84b47e248dba941351896230d9ec`). Since `spacy`has a functions exactly for this kind of situation, that is the way it is built.

## 2) [`dataset_balancing`](data_balancing.py)
Since the problem that the model that is being developed at this repository is a [multilabel classification problem](https://en.wikipedia.org/wiki/Multi-label_classification), the data balancing is a little bit tricky. Because of this the this step exist only to segmentate this step in a logical way.

Other reason is that the training data is collected by hand (since in order to obtain it, one must be logged at scopus and download it locally). Thus, if previously to the execution of this entrypoint, the data was stored on a arbitrary data storage, it is possible, by implementing `load_data()` on a class that inherits from [`SDGloader()`](utils/data_loader.py) to allow reading the data from another source than a local folder.

## 3) [`preprocess_and_split`](splitting.py)
This step is the most straightfoward of all. It takes the data logged previously at step `2)` and with `scikit-learn` make the splits for training and testing. Finally it logs the splitted data at `wandb` again.

## 4) [`advanced_preprocessing`](advanced_preprocessing.py)
The final step is responsible for making all the basic NLP pre-training operations at the datasets (more specifically the text one, i.e., `X_train`/`X_valid`/`X_test`). These operations are, in order:
1) Converting text to lowercase
2) Removing special characters
3) Removing numbers (with a whitespace at its side or sides and being only the number)
4) Removing double spaces
5) Removing accents (normalize('NFKD', str) do things like Ç -> C + ̧)
6) Tokenizing text 
7) Removing stopwords
8.1) Lemmatizing (speeding up it with nlp.pipe as suggested by the spacy documentation)
8.2) Stemming (in parallel with joblib)
9) Converting back to np.array
10) Discarding empty sentences

And then logging it to wandb as a proper TensorFlow dataset.

NOTE: The steps `6)`, `7)` and `8.2)` uses the [`joblib`](https://joblib.readthedocs.io/en/latest/) lib to paralellize the execution of the operations. Step `8.1)`, the most time-consuming one, uses `nlp.pipe()` from `spacy` as suggested by the [lib's documentation](https://spacy.io/usage/processing-pipelines#processing).

# A little hint
Since there is a [`MLproject`](MLproject) at this folder, one can execute this as a separeted project from [the one](../MLproject) at the root of the repository. The only problem is that neither $WANDB_PROJECT nor $WANDB_RUN_GROUP are defined, so the runs at `wandb` will be saved without a proper identification. Finally, this considers that you're using the conda environment definit at `./split_data/conda.yml`. So, before running the following `mlflow`commands, do this:

```shell
$ conda env create -f ./split_data/conda.yml
$ conda activate split_data
```

Then run:

    
```shell
# if the user is at root folder '.':
$ mlflow run ./split_data/ -e download_language_models
$ mlflow run ./split_data/ -e dataset_balancing -P quantile=0.88 # changing default value

# if the user is at split_data folder:
$ mlflow run . -e download_language_models
$ mlflow run . -e dataset_balancing -P quantile=0.88 # changing default value

```