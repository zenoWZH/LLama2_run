#!/bin/bash
sudo chown $USER /var/run/docker.sock
docker pull yival/release:benchmark
docker remove inference_benchmark
docker run --gpus all --name inference_benchmark -p 8888:8888 yival/release:benchmark
docker restart -t 20 inference_benchmark
docker exec -d inference_benchmark bash cd ./LLama2_run && git pull origin master