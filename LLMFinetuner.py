# LLMFinetuner.py
import time
#from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
import os
import torch

from trl import SFTTrainer
from accelerate import Accelerator
import asyncio
import gc

class LLMFinetuner:
    def __init__(self, model, tokenizer, access_token=None, batch_size=4, log_file="./log_temp.txt", dataset=None, **training_args):
        # Output directory where the model predictions and checkpoints will be stored
        self.model = model
        self.batch_size = batch_size
        self.log_filename = log_file
        self.trainer = None
        if dataset:
            self.split_dataset(dataset)
        else:
            self.dataset = None
        self.step=0
        self.tokenizer = tokenizer
        self.max_seq_length = self.tokenizer.model_max_length
        self.training_time = 0
        x = 1
        while(self.max_seq_length > 1):
            x *= 2
            self.max_seq_length = self.max_seq_length // 2
        self.max_seq_length = x
        
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
        return self.dataset
        
    def tune_all_multiGPU(self, peft_config, training_arguments, packing, max_seq_length, dataset_text_field):
        if not self.dataset:
            raise RuntimeError("Dataset not loaded")
        
        start_time = time.time()
        # Set supervised fine-tuning parameters
        try:
            training_arguments.deepspeed = "ds_config.json"
            accelerator = Accelerator()
            print("Accelerator initialized, using all GPUs for training")
            self.trainer = SFTTrainer(
                model=self.model,
                peft_config=peft_config,
                train_dataset=self.dataset['train'],
                eval_dataset=self.dataset['test'],
                args=training_arguments,
                max_seq_length=max_seq_length,
                tokenizer=self.tokenizer,
                packing=packing,
                dataset_text_field=dataset_text_field,
            )
            accelerator.prepare(self.trainer)        
            self.trainer.train()
            self.training_time = time.time() - start_time
            self._log_time('Training time', self.training_time)
            print("\n")
            print(f"Training Whole Part Complete at batch={str(self.batch_size)}")
            #del self.trainer
            gc.collect()
        except BaseException as err:
            print(err)
            gc.collect()
            raise RuntimeError(err)
    
    def tune_step(self, formatted_dataset, peft_config=None, training_arguments=None, packing=None, max_seq_length=None, dataset_text_field=None):
        start_time = time.time()
        # Set supervised fine-tuning parameters
        #
        self.split_dataset(formatted_dataset)
        max_seq_len_trainer = min(self.max_seq_length, max_seq_length)
        self.batch_size = training_arguments.per_device_train_batch_size
        #training_arguments.save_strategy = "epoch"
        training_arguments.save_total_limit = 2
        try:
            #if self.step==0:
            #print("Start from New SFTTrainer")               
            self.trainer = SFTTrainer(
                            model=self.model,
                            peft_config=peft_config,
                            train_dataset=self.dataset['train'],
                            eval_dataset=self.dataset['test'],
                            args=training_arguments,
                            max_seq_length=max_seq_len_trainer,
                            tokenizer=self.tokenizer,
                            packing=packing,
                            dataset_text_field=dataset_text_field,
                        )
        except BaseException as err:
            gc.collect()
            print(err)
            raise RuntimeError("Error in setting up Trainer, or no Trainer")

        try:
            self.step+=1
            self.trainer.train()
            self.model = self.trainer.model
            self.tokenizer = self.trainer.tokenizer
        except BaseException as err:
            gc.collect()
            print(err)
            raise RuntimeError("Error in training model with shard data")

        
        del self.trainer
        gc.collect()
        
        self.training_time += time.time() - start_time
        
        #return self.model, self.tokenizer

    def tune_shard(self, formatted_dataset, peft_config, training_arguments, packing, max_seq_length, dataset_text_field, num_shards=1):
        start_time = time.time()
        max_seq_len_trainer = min(self.max_seq_length, max_seq_length)
        self.batch_size = training_arguments.per_device_train_batch_size
        training_arguments.save_total_limit = 2

        for i in range(num_shards):
            try:
                sub_dataset = formatted_dataset.shard(num_shards, i, keep_in_memory=True, contiguous=True)
                print("Dataset sharded, size is: ", len(sub_dataset))
                self.split_dataset(sub_dataset)
                self.trainer = SFTTrainer(
                                model=self.model,
                                peft_config=peft_config,
                                train_dataset=self.dataset['train'],
                                eval_dataset=self.dataset['test'],
                                args=training_arguments,
                                max_seq_length=max_seq_len_trainer,
                                tokenizer=self.tokenizer,
                                packing=packing,
                                dataset_text_field=dataset_text_field,
                            )
            except BaseException as err:
                gc.collect()
                print(err)
                raise RuntimeError("Error in setting up Trainer, or no Trainer")

            try:
                self.step+=1
                self.trainer.train()
                self.model = self.trainer.model
                self.tokenizer = self.trainer.tokenizer
            except BaseException as err:
                gc.collect()
                print(err)
                raise RuntimeError("Error in training model with shard data")

            
            del self.trainer
            gc.collect()
        
        self.training_time += time.time() - start_time