#!/bin/bash
sudo chown $USER /var/run/docker.sock
docker pull yival/release:benchmark
docker stop inference_benchmark
docker remove inference_benchmark
docker run --gpus all --name inference_benchmark -p 8888:8888 -d yival/release:benchmark /bin/bash -c "while true; do sleep 1; done"