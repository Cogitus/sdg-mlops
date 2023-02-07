from pathlib import Path
from tempfile import TemporaryDirectory

import nltk
from utils.data_loader import LocalSDGLoader


def main() -> None:
    # current_dir = Path.cwd()

    data_loader = LocalSDGLoader()

    with TemporaryDirectory() as tmp_dir:
        # Set the location of the NLTK data directory to a custom folder
        nltk_data_folder = Path(tmp_dir) / "nltk_data"

        nltk.download("punkt", download_dir=nltk_data_folder)
        nltk.download("stopwords", download_dir=nltk_data_folder)

        arquivos = data_loader.load_data(
            data_location=Path(
                "/home/alsinaariel/Downloads/SDGs-20230122T202630Z-001/SDGs"
            )
        )

        print(arquivos)


if __name__ == "__main__":
    main()
