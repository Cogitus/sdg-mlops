.ONESHELL:

include .env
export

SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
CONDA_ENV_NAME=gradio_testing
PYTHON_VERSION=3.10
DOCKER_IMAGE_GRADIO=gradio-model

.PHONY: create_conda_env
create_env:
	@echo "Creating the $(CONDA_ENV_NAME) conda python environment for local runs"
	@conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION) -y
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME)
	@echo "Installing the pip packages at `requirements.txt`"
	@pip3 install -r requirements.txt

.PHONY: deploy_local
deploy_local:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME)
	python3 model.py

.PHONY: build
build:
	docker build --tag $(DOCKER_IMAGE_GRADIO) .

.PHONY: deploy
deploy:
	docker run \
		--env AWS_ACCESS_KEY_ID=$$AWS_ACCESS_KEY_ID \
		--env AWS_SECRET_ACCESS_KEY=$$AWS_SECRET_ACCESS_KEY \
		--env MLFLOW_TRACKING_URI=$$MLFLOW_TRACKING_URI \
		--rm -p 7860:7860 --detach \
		--name gradio-model-server $(DOCKER_IMAGE_GRADIO):latest

.PHONY: clean
clean:
	@echo "Removing the local $(CONDA_ENV_NAME) conda python environment"
	conda remove --name $(CONDA_ENV_NAME) --all -y
	@echo "Deleting the Gradio folder flagged/"
	rm -rf flagged

# .PHONY: remove_docker_container
# remove_docker_container:

reset: clean