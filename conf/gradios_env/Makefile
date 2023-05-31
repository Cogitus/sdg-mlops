.ONESHELL:

SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
CONDA_ENV_NAME=gradio_testing
PYTHON_VERSION=3.10

.PHONY: deploy
deploy:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME)
	python3 model.py

.PHONY: create_env
create_env:
	conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION)
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME)
	pip install -r requirements.txt

.PHONY: clean
clean:
	conda remove --name $(CONDA_ENV_NAME) --all
	rm -rf flagged