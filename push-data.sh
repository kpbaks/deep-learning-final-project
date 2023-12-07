#!/bin/bash

# push a path to a dataset on remote

path=$1; shift
dataset=$1; shift

function usage() {
    echo "ERROR: wrong arguments"
    echo "usage:"
    echo "push-data.sh LOCAL_PATH DATASET_FOLDER"
}

if [ -z ${path} ]; then usage; exit 1; fi
if [ -z ${dataset} ]; then usage; exit 1; fi

rsync -r "${path}/" "dl25e23@gpucluster.st.lab.au.dk:~/datasets/${dataset}/"
