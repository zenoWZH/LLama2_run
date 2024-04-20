
# Benchmark Script to test GPU fine-tuning LLMs

* Step 1: Run dockersetup.sh to pull the docker image of Yival to get enviorment
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