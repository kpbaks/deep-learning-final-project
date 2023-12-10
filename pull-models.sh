model_name=$1; shift
epoch=$1; shift

function usage() {
    echo "ERROR: wrong arguments"
    echo "usage:"
    echo "pull-data.sh MODEL_NAME EPOCH"
}

if [ -z ${model_name} ]; then usage; exit 1; fi
if [ -z ${epoch} ]; then usage; exit 1; fi

model_full_name="${model_name}/epoch_${epoch}"
remote_path="dl25e23@gpucluster.st.lab.au.dk:deep-learning-final-project/models/${model_full_name}/*"
local_path="models/${model_full_name}/"

mkdir -p "$local_path"
scp -r "$remote_path" "$local_path"
