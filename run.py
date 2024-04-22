from ConfigReader import ConfigReader
import os
import time
import sys
from tqdm import tqdm
import torch

def retry_finetuning(model, dataset, batch_size=4):
    torch.cuda.empty_cache()
    
    print('='*80)
    print("\n")
    print(f"Try again with batch_size = {batch_size}")
    print('='*80)
    print("\n")
    syscode = os.system(f"poetry run python main.py {model} {dataset} {str(batch_size)}")
    if syscode == 0:
            print("Training Successful!!!")
    else:
        torch.cuda.empty_cache()
        if batch_size<=1:
            print("Aborting fine-tuning of this model and dataset")
        else:
            retry_finetuning(model, dataset, batch_size//2)
        
print(__name__)
if __name__ == "__main__":
    print("RUNNING!!!")
    #print("\n")
    model_reader = ConfigReader("models.txt")
    dataset_reader = ConfigReader("datasets.txt")
    models = model_reader.read_lines_without_comments()
    datasets = dataset_reader.read_lines_without_comments()
    batch_size = 8
    for model in tqdm(models, desc="Models"):
        for dataset in tqdm(datasets, desc="Datasets", leave=False):
            os.system('clear')
            print('='*80)
            print("\n")
            print(f"Training on model {model} with dataset {dataset}")
            print("Start with batch_size = 8\n")
            print('='*80)
            print("\n")
            syscode = os.system(f"poetry run python main.py {model} {dataset} {str(batch_size)}")
            if syscode == 0:
                print("Training Successful!!!")
            else:
                torch.cuda.empty_cache()
                retry_finetuning(model, dataset, batch_size//2)

            time.sleep(5)
        