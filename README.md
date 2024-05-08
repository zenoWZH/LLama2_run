
# Benchmark Script to test GPU fine-tuning LLMs
Make sure your server is installed with docker and nvidia toolkits for docker, and your user is sudoer of your system

## All in one command:

* Step 1: Pull this repo down to your server
* Step 2: Run the bash script
```
    bash docker_run.sh
```
or in branch shard_data
```
    bash docker_run_shard.sh
```

## Run each step by yourself

* Step 1: Pull the docker image of Yival to get enviorment
```
    docker pull yival/release:benchmark
```
* Step 2: Run docker image interactively
```
    docker run --gpus all -it -p 8888:8888 yival/release:bechmark
```
* Step 3: Enter the LLama2_run.git and setup the enviorment
```
    git clone https://github.com/zenoWZH/LLama2_run.git
    cd LLama2_run
    poetry install
```
* Step 4: run main.py under poetry enviorment
```
    poetry run python main.py
```
* Step 5: open log in txt file and record your loading time, training time and total time
