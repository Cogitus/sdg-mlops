#!/bin/bash

# get the list of all envs that starts with `mflow` the tells conda
# to unninstall then.
for conda_enviroment in $(conda env list | grep mlflow | cut -f1 -d" "); do
    conda uninstall -y --name $conda_enviroment --all
done