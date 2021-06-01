docker run \
    --gpus all \
    --shm-size 32g \
    --volume ${PWD}/src:/home/src:Z \
    --volume ${PWD}/results:/home/results:Z \
    -ti passage2:latest "$@"
