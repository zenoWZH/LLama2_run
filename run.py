from ConfigReader import ConfigReader
import os
import time
import sys
from tqdm import tqdm
import torch

def retry_finetuning(model, dataset, batch_size=4):
    torch.cuda.empty_cache()
    try:
        print('='*80)
        print("\n")
        print(f"Try again with batch_size = {batch_size}")
        print('='*80)
        print("\n")
        os.system(f"poetry run python main.py {model} {dataset} {batch_size}")
        print("Training Successful!!!")
        torch.cuda.empty_cache()
        time.sleep(5)
        
    except RuntimeError as err:
        if batch_size<=1:
            print(err)
            print("Aborting fine-tuning of this model and dataset")
            torch.cuda.empty_cache()
        else:
            retry_finetuning(model, dataset, batch_size//2)
print(__name__)
if __name__ == "__main__":
    print("RUNNING!!!")
    print("\n")
    model_reader = ConfigReader("models.txt")
    dataset_reader = ConfigReader("datasets.txt")
    models = model_reader.read_lines_without_comments()
    datasets = dataset_reader.read_lines_without_comments()
    batch_size = 8
    for model in tqdm(models, desc="Models"):
        for dataset in tqdm(datasets, desc="Datasets", leave=False):
            os.system('clear')
            try:
                print('='*80)
                print("\n")
                print(f"Training on model {model} with dataset {dataset}")
                print("Start with batch_size = 8\n")
                print('='*80)
                print("\n")
                
                print("Training Successful!!!")
                torch.cuda.empty_cache()
                time.sleep(5)
            except RuntimeError as err:
                #print(err)
                print("GPU OUT OF MEMORY!!! Retrying fine-tuning after clear cache")
                retry_finetuning(model, dataset, batch_size//2)
                torch.cuda.empty_cache()
                continue