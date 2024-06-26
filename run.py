from ConfigReader import ConfigReader
import os
import time
import sys
from tqdm.auto import tqdm
import torch
import shutil
import gc

def log_info(message, log_file= "templog.txt"):
    print(message.strip())
    with open(log_file, "a+") as file:
        file.write(message)
        file.write("\n")
    
def clear_cache():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.cuda.empty_cache()
    cache_dir = os.path.join(os.getcwd(), '.cache', 'huggingface')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    gc.collect()
    print("Cache cleared") 

def retry_finetuning(model, dataset, batch_size=4):
    print('='*80)
    print("\n")
    message = f"Try model {model.split('/')[-1]} on dataset {dataset.split('/')[-1]} again with batch_size = {batch_size}"
    log_info(message, "./runninglog.txt")
    print('='*80)
    print("\n")
    torch.cuda.empty_cache()
    syscode = os.system(f"poetry run python main.py {model} {dataset} {str(batch_size)}")
    message = f"*************SYSTEM EXIT WITH CODE {syscode}*********************"
    log_info(message, "./runninglog.txt")
    if syscode == 0:
        print("Training Successful!!!")
    else:
        torch.cuda.empty_cache()
        if batch_size<=1:
            print("Aborting fine-tuning of this model and dataset")
        else:
            retry_finetuning(model, dataset, batch_size//2)
        

if __name__ == "__main__":
    print("RUNNING!!!")
    #print("\n")
    model_reader = ConfigReader("models.txt")
    dataset_reader = ConfigReader("datasets.txt")
    models = model_reader.read_lines_without_comments()
    datasets = dataset_reader.read_lines_without_comments()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    batch_size = 4
    for model in tqdm(models, desc="Models"):
        for dataset in tqdm(datasets, desc="Datasets", leave=False):
            os.system('clear')
            gc.collect()
            print('='*80)
            print("\n")
            message = f"Training on model {model.split('/')[-1]} with dataset {dataset.split('/')[-1]} in batch_size = {batch_size}"
            log_info(message, "./runninglog.txt")
            print('='*80)
            print("\n")
            syscode = os.system(f"poetry run python main.py {model} {dataset} {str(batch_size)}")
            if syscode == 0:
                print("Training Successful!!!")
            else:
                clear_cache()
                retry_finetuning(model, dataset, batch_size)
            time.sleep(5)
        
