import logging
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SDGloader(ABC):
    wandb_project: str = "sdg-onu"

    def __init__(self) -> None:
        self._is_balanced = False
        self._datasets = None  # will store all 16 SDG dataframe/datasets

    # a "wrapper" to make it work only after the data processing
    def import_data(self, data_path: Path) -> None:
        if not self._is_balanced:
            raise RuntimeError(
                f"{self.__class__.__name__} must be balanced before importing"
            )

        # w&b importing goes here

    def _binarize_data(self, datasets: pd.DataFrame) -> None:
        mlb = MultiLabelBinarizer()

        for i, label_dataset in enumerate(datasets):
            targets = mlb.fit_transform(
                label_dataset["Sustainable Development Goals (2021)"]
                .str.replace(" ", "")
                .str.split("|")
            )

            targets_dataframe = pd.DataFrame(
                targets, columns=mlb.classes_, dtype=np.float32
            )

            # the concatenation is made horizontally along the x-axis (columns)
            datasets[i] = pd.concat([datasets[i], targets_dataframe], axis="columns")
            # remove the intermediary column
            datasets[i] = datasets[i].drop(
                columns=["Sustainable Development Goals (2021)"]
            )

    def remove_duplicates(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self._is_balanced:
            raise RuntimeError(
                f"{self.__class__.__name__} the removal of duplicates was already done"
            )
        else:
            # remove duplicate data
            dataset = dataset.drop_duplicates()

            # perform union set on labels for duplicated text entries
            # but different target sets
            text_data = dataset[["text"]].copy()
            SDG_COLUMNS = [col for col in dataset.columns if col.startswith("SDG")]

            title_counts = text_data["text"].value_counts()
            duplicated_titles = title_counts[title_counts > 1].index.tolist()

            cleaned_title_rows = []
            for title in duplicated_titles:
                sdg_column_data = dataset[SDG_COLUMNS].loc[dataset["text"] == title, :]
                # this binarizes the presence of the SDGs (columns) for this iteration of `title`
                sdg_binarization = sdg_column_data.sum(axis="index") > 0
                sdg_binarization = sdg_binarization.astype(int).tolist()

                # recreates a dataframe line as a list()
                agg_data = [title]
                agg_data.extend(sdg_binarization)

                cleaned_title_rows.append(agg_data)

            deduplicated_records = pd.DataFrame(
                cleaned_title_rows, columns=["text"] + SDG_COLUMNS
            )

            deduplicated_dataset = dataset.loc[~dataset["text"].isin(duplicated_titles)]
            deduplicated_dataset = pd.concat(
                [deduplicated_dataset, deduplicated_records], ignore_index=True
            )

            return deduplicated_dataset

    # the methods that do the proper loading and importing of the data
    @abstractmethod
    def load_data(
        self, data_location: Path | list[str], persist_data: bool = False
    ) -> Path | list[pd.DataFrame]:
        ...

    # getters of the private attributes (read_only)
    @property
    def datasets(self) -> list[pd.DataFrame]:
        return self._datasets

    @property
    def is_balanced(self) -> bool:
        return self._is_balanced


class LocalSDGLoader(SDGloader):
    def __init__(self) -> None:
        super().__init__()

    def load_data(
        self, data_location: Path | list[str], persist_data: bool = False
    ) -> Path | list[pd.DataFrame]:
        logger.info(f"loading sdg files from {data_location}")

        files = glob(str(data_location) + "/*.csv")
        files = sorted(files)

        logger.info(f"the following files were loaded to memory:")

        datasets = []
        for file in files:
            logger.info(f"\t{file}")
            datasets.append(pd.read_csv(file, sep="\t"))

        # label encoding with MultiLabelBinarizer()
        self._binarize_data(datasets)

        # the concatenation is made vertically, along the y-axis (rows)
        data = pd.concat(datasets, ignore_index=True, axis="index")
        data = data.rename(columns={"Title": "text"})

        if persist_data is False:
            data = self.remove_duplicates(data)
            self._datasets = data

            return data


class WandbSDGLoader(SDGloader):
    def __init__(self) -> None:
        super().__init__()

    def load_data(
        self, data_location: Path | list[str], persist_data: bool = False
    ) -> Path | list[pd.DataFrame]:
        pass


class GithubSDGLoader(SDGloader):
    def __init__(self) -> None:
        super().__init__()

    def load_data(
        self, data_location: Path | list[str], persist_data: bool = False
    ) -> Path | list[pd.DataFrame]:
        pass


if __name__ == "__main__":
    data_loader = LocalSDGLoader()
    data = data_loader.load_data(data_location=Path("/home/alsinaariel/Downloads/SDGs"))

    # data.to_csv("APAGAR.csv")
