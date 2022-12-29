import json
import logging
import os
import tempfile
import warnings
from functools import partial
from typing import Any, Optional

import requests
import wandb
from bs4 import BeautifulSoup
from progressbar import ETA, Bar, Percentage, ProgressBar, SimpleProgress, Timer
from urllib3.exceptions import InsecureRequestWarning

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s", datefmt="%d/%m/%Y %H:%M:%S"
)

# filters the verbose of warning logs incoming from the SSL errors
warnings.filterwarnings("ignore", category=InsecureRequestWarning)


def collect_documents(
    landing_page_url: str,
    titles: list,
    authors: list,
    affiliations: list,
    dois: list,
    keywords: list,
    abstracts: list,
) -> None:
    # supressing the warnings in case of requests fails
    warnings.filterwarnings("ignore", category=InsecureRequestWarning)

    PBAR_WIDGETS = [
        "Processing pages: ",
        Percentage(),
        " (",
        SimpleProgress(),
        ") ",
        Bar(marker="●", fill="○"),
        " ",
        ETA(),
        " ",
        Timer(),
    ]

    logger.info(f"Getting lading page of CBA at '{landing_page_url}'")
    html_content = requests.get(landing_page_url, verify=False, timeout=30).text
    landing_page = BeautifulSoup(html_content, "html.parser")

    paper_pages = []
    obj_article = landing_page.body.find_all(
        "div", attrs={"class": "obj_article_summary"}
    )

    # getting all the articles pages (where their data is stored)
    for obj in obj_article:
        paper_pages.append(obj.find("div", attrs={"class": "title"}).a["href"])
    paper_pages = paper_pages[3:]

    N_PAGES = len(paper_pages)

    # progress bar for visualization of the progress
    pbar = ProgressBar(maxval=N_PAGES, widgets=PBAR_WIDGETS, redirect_stdout=True)
    pbar.start()

    for i, page_url in enumerate(paper_pages):
        proceed = False

        # the while is need since it will only allows a new iteration if
        # the previous one was a success
        while not proceed:
            try:
                html_content = requests.get(page_url, verify=False, timeout=30).text
            except requests.exceptions.ConnectionError:
                logger.error(f"An error occurred at the request of {page_url}")
            else:
                page = BeautifulSoup(html_content, "html.parser")
                proceed = True

        # get paper title
        current_title = page.find("h1", class_="page_title").text.strip()
        titles.append(current_title)

        # get author names
        author_list = page.body.find(
            "article", attrs={"class": "obj_article_details"}
        ).find_all("span", attrs={"class": "name"})

        current_authors_list = []
        for author_span in author_list:
            current_authors_list.append(author_span.text.strip())
        authors.append(current_authors_list)

        # get author affiliations
        affiliations_list = page.body.find(
            "article", attrs={"class": "obj_article_details"}
        ).find_all("span", attrs={"class": "affiliation"})

        current_affiliations_list = []
        for affiliation_span in affiliations_list:
            current_affiliations_list.append(affiliation_span.text.strip())
        affiliations.append(current_affiliations_list)

        # get doi
        current_doi = (
            page.body.find("article", attrs={"class": "obj_article_details"})
            .find("div", class_="item doi")
            .a.string.strip()
        )
        dois.append([current_doi])

        # get keywords
        current_keywords = (
            page.body.find("article", attrs={"class": "obj_article_details"})
            .find("div", class_="item keywords")
            .find("span", attrs={"class": "value"})
            .text.split(",")
        )
        current_keywords = [keyword.strip() for keyword in current_keywords]
        keywords.append(current_keywords)

        # get abstract
        current_abstract = [page.find("div", class_="item abstract").p.text.strip()]
        abstracts.append(current_abstract)

        # update progress bar
        pbar.update(i)

    pbar.finish()


def save_data(
    data: Any,
    project_name: str,
    filename: str,
    filepath: Optional[str] = os.getcwd(),
    remote: Optional[bool] = False,
) -> None:
    # local where the file already is or where is to save it (locally)
    save_path = os.path.join(filepath, filename)

    if remote:
        logger.info(f"Saving {filename} at a wandb project `{project_name}`")

        with tempfile.TemporaryDirectory() as TMP_DIR:
            logger.info("Starting conection with WandB")

            with wandb.init(
                job_type="download_data",
                project="sdg-onu",
                tags=["dev", "data", "download"],
            ) as run:
                logger.info(f"Creating artifact for {filename} at {TMP_DIR}")

                # instatiating a new artifact for the data
                artifact = wandb.Artifact(
                    name=filename, type="raw_data", description=""
                )

                try:
                    # contents of the file for 0 is '' and 2 is '[]'
                    is_empty_file = os.path.getsize(save_path) in [0, 2]
                except FileNotFoundError:
                    is_empty_file = True

                # conditions for writing a file on the temporary folder:
                # the downloaded data must not be already saved locally.
                if not os.path.exists(save_path) or is_empty_file:
                    # modifying the value of `save_path` to one in a tmp folder
                    save_path = os.path.join(TMP_DIR, filename)
                    with open(save_path, "wt") as file_handler:
                        json.dump(data, file_handler, indent=2)

                artifact.add_file(save_path, filename)

                logger.info(f"Logging `{filename}` artifact.")

                run.log_artifact(artifact)
                artifact.wait()
    else:
        logger.info(f"Saving {filename} at {filepath}")

        with open(save_path, "w") as file:
            json.dump(data, file, indent=2)


def main() -> None:
    # fields to fill with the scrapped data
    titles = []
    authors = []
    affiliations = []
    dois = []
    keywords = []
    abstracts = []

    # simplifying the call of the function that realizes the webscraping
    get_data = partial(
        collect_documents,
        titles=titles,
        authors=authors,
        affiliations=affiliations,
        dois=dois,
        keywords=keywords,
        abstracts=abstracts,
    )

    get_data("https://www.sba.org.br/open_journal_systems/index.php/cba")

    wandb_save = partial(save_data, project_name="sdg-onu", remote=True)

    wandb_save(data=titles, filename="titles.json")
    wandb_save(data=authors, filename="authors.json")
    wandb_save(data=affiliations, filename="affiliations.json")
    wandb_save(data=dois, filename="dois.json")
    wandb_save(data=keywords, filename="keywords.json")
    wandb_save(data=abstracts, filename="abstracts.json")


if __name__ == "__main__":
    main()
