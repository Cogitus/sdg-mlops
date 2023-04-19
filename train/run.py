import logging

from utils.metrics import *

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    logger.info("it worked")


if __name__ == "__main__":
    main()
