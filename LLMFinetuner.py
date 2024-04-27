# LLMFinetuner.py
import time
#from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import os
import torch

from trl import SFTTrainer
import asyncio
import gc

class LLMFinetuner:
    def __init__(self, model, access_token=None, batch_size=4, log_file="./temptest.txt", **training_args):
        # Output directory where the model predictions and checkpoints will be stored
        self.model = model
        self.batch_size = batch_size
        self.log_filename = log_file
        self.trainer = None
        self.dataset = None
    
    def __del__(self):
        try:
            del self.trainer
            del self.dataset
        except:
            pass
        gc.collect()
        #sys.exit(0) vs os._exit(0)
    
    def _log_time(self, prefix, seconds):
        # Calculate time format
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        time_str = "{:0>1} days, {:0>2} hours, {:0>2} minutes, {:0>2} seconds".format(int(days), int(hours), int(minutes), int(secs))
        message = f'{prefix}: {time_str}\n'

        print(message.strip())
        with open(self.log_filename, "a+") as file:
            file.write(message)
            
    def split_dataset(self, formatted_dataset):
        if formatted_dataset is not None and not ('train' in formatted_dataset.column_names):
            # 数据集通常包含训练和测试集，但如果需要自定义比例，可以使用以下方法
            train_test_split = formatted_dataset.train_test_split(test_size=0.1)

            # 创建一个新的数据集字典，包括训练集和测试集
            self.dataset = DatasetDict({
                'train': train_test_split['train'],
                'test': train_test_split['test']
            })
        elif not('test' in formatted_dataset.column_names):
            train_test_split = formatted_dataset['train'].train_test_split(test_size=0.1)
            self.dataset = DatasetDict({
                'train': train_test_split['train'],
                'test': train_test_split['test']
            })
        else:
            self.dataset = formatted_dataset
        
    def tune(self, peft_config, training_arguments, tokenizer, packing, max_seq_length, dataset_text_field):
        if not self.dataset:
            raise RuntimeError("Dataset not loaded")
        
        start_time = time.time()
        # Set supervised fine-tuning parameters
        try:
            self.trainer = SFTTrainer(
                model=self.model,
                peft_config=peft_config,
                train_dataset=self.dataset['train'],
                eval_dataset=self.dataset['test'],
                args=training_arguments,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer,
                packing=packing,
                dataset_text_field=dataset_text_field,
            )
        
            self.trainer.train()
            self.training_time = time.time() - start_time
            self._log_time('Training time', self.training_time)
            print("\n")
            print(f"Training Whole Part Complete at batch={str(self.batch_size)}")
            #del self.trainer
            gc.collect()
        except BaseException as err:
            gc.collect()
            raise RuntimeError(err)
    
    def tune_step(self, formatted_dataset, peft_config=None, training_arguments=None, tokenizer=None, packing=None, max_seq_length=None, dataset_text_field=None):
        self.start_time = time.time()
        # Set supervised fine-tuning parameters
        #
        self.split_dataset(formatted_dataset)
        try:
            self.trainer = SFTTrainer(
                            model=self.model,
                            peft_config=peft_config,
                            train_dataset=self.dataset['train'],
                            eval_dataset=self.dataset['test'],
                            args=training_arguments,
                            max_seq_length=max_seq_length,
                            tokenizer=tokenizer,
                            packing=packing,
                            dataset_text_field=dataset_text_field,
                        )
        except BaseException as err:
            gc.collect()
            print("Error in setting up Trainer, or no Trainer")
            raise RuntimeError(err)

        del self.dataset
        gc.collect()
        print("\n")
        print(f"Training Complete at batch={str(self.batch_size)} in shattered mode")
        self.training_time = time.time() - self.start_time
        self._log_time('Training time', self.training_time)
        return self.model
