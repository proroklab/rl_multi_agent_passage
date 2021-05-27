docker run \
    --gpus '"device=0"' \
    --shm-size 32g \
    --volume ${PWD}/src:/home/src:Z \
    --volume ${PWD}/results:/home/results:Z \
    -ti marlobjtrack:latest "$@"
