# main.py
from LLMFinetuner import LLMFinetuner

import os
import sys
from tqdm import tqdm
import gc
import torch
import shutil
import time
import asyncio
from ConfigReader import ConfigReader
import warnings


def clear_cache():
    torch.cuda.empty_cache()
    cache_dir = os.path.join(os.getcwd(), '.cache', 'huggingface')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print("Cache cleared")   



print(__name__)
#if __name__ == "__main__":
print("START MAIN PROCESS!!!")
print("\n")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
warnings.filterwarnings("ignore")
default_access_token = ConfigReader("access_token.txt").read_lines_without_comments()[0]

if len(sys.argv)>5 or len(sys.argv)<4:
    print("Usage: python main.py <model_name> <dataset_name> <batch_size> <access_token>")
    sys.exit(1)

elif len(sys.argv)>=4:
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    batch_size = int(sys.argv[3])
    
    if len(sys.argv)==4:
        access_token = default_access_token
    else:
        access_token = sys.argv[4]    

try:
    finetuner = LLMFinetuner(model_name, dataset_name, access_token, batch_size)
    finetuner.train()
except BaseException as err:
    print('='*80)
    print("ERROR!!!\n")
    print(err)
    print('='*80)
    print("\n")
    sys.exit(1)
del finetuner
gc.collect()
sys.exit(0)
                
