model_name=$1; shift

function usage() {
    echo "ERROR: wrong arguments"
    echo "usage:"
    echo "pull-data.sh MODEL_NAME"
}

if [ -z ${model_name} ]; then usage; exit 1; fi

scp -r dl25e23@gpucluster.st.lab.au.dk:deep-learning-final-project/models/${model_name}/* "models/${model_name}/"
