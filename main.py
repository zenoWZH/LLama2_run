# main.py
from LLMFinetuner import LLMFinetuner
from ConfigReader import ConfigReader

import os
import sys
from tqdm import tqdm
import gc
import torch
import shutil
import time
import asyncio
import warnings

class Finetuner:
    def __init__(self, model_name, dataset_name, access_token, batch_size):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.access_token = access_token
        self.batch_size = batch_size

    def finetune(self):
        finetuner = LLMFinetuner(self.model_name, self.dataset_name, self.access_token, self.batch_size)
        finetuner.train()
        del finetuner
        gc.collect()
        return 0
    
    def main_training(model_name, dataset_name, batch_size, access_token):
        print("START MAIN PROCESS!!!")
        print("\n")
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

print(__name__+" is running!!!")                
if __name__ == "__main__":
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

    main_training(model_name, dataset_name, batch_size, access_token)