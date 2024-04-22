# main.py
from LLMFinetuner import LLMFinetuner
from ConfigReader import ConfigReader
import os
import sys
from tqdm import tqdm
import gc
import torch
import shutil

def clear_cache():
    torch.cuda.empty_cache()
    cache_dir = os.path.join(os.getcwd(), '.cache', 'huggingface')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print("Cache cleared")

def retry_finetuning(model, dataset, access_token, batch_size=8):
    try:
        finetuner = LLMFinetuner(model, dataset, access_token, batch_size=batch_size)
        finetuner.train()
    except RuntimeError as err:
        del finetuner
        clear_cache()
        gc.collect()
        if batch_size<=1:
            print(err)
            print("Aborting fine-tuning of this model and dataset")
        else:
            retry_finetuning(model, dataset, access_token, batch_size/2)
        
        
default_access_token = ConfigReader("access_token.txt").read_lines_without_comments()[0]
if __name__ == "__main__":
    if len(sys.argv) < 4 and len(sys.argv) > 1:
        print("Usage: python main.py <model_name> <dataset_name> <access_token>")
        sys.exit(1)
    elif len(sys.argv) == 4:
        model_name = sys.argv[1]
        dataset_name = sys.argv[2]
        access_token = sys.argv[3]
        
        finetuner = LLMFinetuner(model_name, dataset_name, access_token, per_device_train_batch_size=8, num_train_epochs=3)
        finetuner.train()
    else:
        access_token = default_access_token
        model_reader = ConfigReader("models.txt")
        dataset_reader = ConfigReader("datasets.txt")
        models = model_reader.read_lines_without_comments()
        datasets = dataset_reader.read_lines_without_comments()
        for model in tqdm(models, desc="Models"):
            for dataset in tqdm(datasets, desc="Datasets", leave=False):
                try:
                    print('='*80)
                    print("\n")
                    print("Start with batch_size=16\n")
                    finetuner = LLMFinetuner(model, dataset, access_token, batch_size=16)
                    finetuner.train()
                    gc.collect()
                except RuntimeError as err:
                    print(err)
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                    del finetuner
                    clear_cache()
                    gc.collect()
                    print("Retrying fine-tuning after clear cache")
                    retry_finetuning(model, dataset, access_token)
                    continue
