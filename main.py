# main.py
from LLMFinetuner import LLMFinetuner
from ConfigReader import ConfigReader
from DatasetFormatter import DatasetFormatter

from datasets import load_dataset, DatasetDict, concatenate_datasets
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)

import os
import sys
from tqdm import tqdm
import gc
import torch
import shutil
import time
import warnings

class FinetuneLoader:
    def __init__(self, model_name, dataset_name, access_token, batch_size, output_dir="./results/checkpoints", epochs=1):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.access_token = access_token
        self.batch_size = int(batch_size)
        self.output_dir = output_dir
        self.training_epochs = epochs
        # Use temporary variables for filename
        model_short_name = self.model_name.split('/')[-1]
        dataset_short_name = self.dataset_name.split('/')[-1]
        if not os.path.exists("./results/logs/"):
            os.makedirs("./results/logs")
        self.logfile = f"./results/logs/{model_short_name}_{dataset_short_name}_batch{self.batch_size}_epochs{self.training_epochs}.log"
        self.shattered_logfile = f"./results/logs/{model_short_name}_{dataset_short_name}_batch{self.batch_size}_epochs{self.training_epochs}_shard.log"
        self.output_file = f"{output_dir}+{model_short_name}_{dataset_short_name}_batch{self.batch_size}_epochs{self.training_epochs}"
    
    def _log_time(self, prefix, seconds, log_file=None):
        if not log_file:
            log_file = self.logfile
        # Calculate time format
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        time_str = "{:0>1} days, {:0>2} hours, {:0>2} minutes, {:0>2} seconds".format(int(days), int(hours), int(minutes), int(secs))
        message = f'{prefix}: {time_str}\n'

        print(message.strip())
        with open(log_file, "a+") as file:
            file.write(message)
    
    def load_dataset(self, iterable=False):
        # Load dataset (you can process it here)
                
        self.dataset_loader = DatasetFormatter(self.dataset_name, iterable=iterable)
        self.dataset_loader.load_dataset()
        ################################################################################
        # Load the dataset
        start_time = time.time()
        self.formatted_dataset = self.dataset_loader.format_dataset()
        # Load LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=access_token, padding="max_length", truncation=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
        #self.tokenized_dataset = self.formatted_dataset.map(lambda examples: self.tokenizer(examples["text"]), batched=True)
        self.data_loading_time = time.time() - start_time
        self._log_time('Dataset preparing time', self.data_loading_time)
        
    def load_model(self):
        ################################################################################
        # QLoRA parameters
        # LoRA attention dimension
        lora_r = 64

        # Alpha parameter for LoRA scaling
        lora_alpha = 16

        # Dropout probability for LoRA layers
        lora_dropout = 0.05

        ################################################################################
        # bitsandbytes parameters
        # Activate 4-bit precision base model loading
        use_4bit = True

        # Compute dtype for 4-bit base models
        bnb_4bit_compute_dtype = "float16"

        # Quantization type (fp4 or nf4)
        bnb_4bit_quant_type = "nf4"

        # Activate nested quantization for 4-bit base models (double quantization)
        use_nested_quant = False

        ################################################################################
        # TrainingArguments parameters
        # Number of training epochs
        num_train_epochs = self.training_epochs

        # Enable fp16/bf16 training (set bf16 to True with an A100)
        fp16 = False
        bf16 = True

        # Batch size per GPU for training
        per_device_train_batch_size = batch_size

        # Batch size per GPU for evaluation
        per_device_eval_batch_size = batch_size

        # Number of update steps to accumulate the gradients for
        gradient_accumulation_steps = 1

        # Enable gradient checkpointing
        gradient_checkpointing = True

        # Maximum gradient normal (gradient clipping)
        max_grad_norm = 0.3

        # Initial learning rate (AdamW optimizer)
        learning_rate = 2e-4

        # Weight decay to apply to all layers except bias/LayerNorm weights
        weight_decay = 0.001

        # Optimizer to use
        optim = "paged_adamw_32bit"

        # Learning rate schedule
        lr_scheduler_type = "cosine"

        # Number of training steps (overrides num_train_epochs)
        max_steps = -1

        # Ratio of steps for a linear warmup (from 0 to learning rate)
        warmup_ratio = 0.03

        # Group sequences into batches with same length
        # Saves memory and speeds up training considerably
        group_by_length = False

        # Save checkpoint every X updates steps
        save_steps = 0

        # Log every X updates steps
        logging_steps = 25

        ################################################################################
        # SFT parameters
        # Maximum sequence length to use
        self.max_seq_length = 4096

        # Pack multiple short examples in the same input sequence to increase efficiency
        self.packing = False

        # Load the entire model on the GPU 0
        device_map = {"": 0}
        # Load the model
        start_time = time.time()
        
        ### Load model with QLoRA configuration
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )
        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            token=access_token
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        # Load LoRA configuration
        self.peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
                ],
        )
        #self.peft_model = get_peft_model(self.model, self.peft_config)
        self.model_loading_time = time.time() - start_time
        self._log_time('Model loading time', self.model_loading_time)
        
        # Set training parameters
        self.training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            logging_dir=self.output_dir+"/logs",
        )

    def finetune_all(self):
        start_time = time.time()
        finetuner = LLMFinetuner(self.model, self.tokenizer, self.batch_size, log_file=self.logfile)
        self.data_loading_time = time.time() - start_time
        self._log_time('Data split time', self.data_loading_time)
        finetuner.tune(self.formatted_dataset,
                        peft_config=self.peft_config, \
                        training_arguments=self.training_arguments, \
                        packing=self.packing, \
                        max_seq_length=self.max_seq_length, \
                        dataset_text_field="text")
        del finetuner
        gc.collect()
        return 0
    
    def finetune_shard_stepped(self, size_per_shard=1024):
        start_time = time.time()
        finetuner = LLMFinetuner(self.model, self.tokenizer, self.batch_size, log_file=self.logfile)
        if len(self.formatted_dataset)<=2:
            if len(self.formatted_dataset)==1:
                self.formatted_dataset = self.formatted_dataset["train"]
            else:
                self.formatted_dataset = concatenate_datasets([self.formatted_dataset["train"], self.formatted_dataset["test"]])
        num_shards = (len(self.formatted_dataset)+size_per_shard-1) // size_per_shard
        print(f"===================Shard dataset {self.dataset_name.split('/')[-1]} into {num_shards} parts==================")
        # 手动数据切片和训练
        try:
            for i in range(0, num_shards):
                sub_dataset = self.formatted_dataset.shard(num_shards, i, keep_in_memory=True, contiguous=True)
                finetuner.tune_step(sub_dataset,
                                    peft_config=self.peft_config, \
                                    training_arguments=self.training_arguments, \
                                    packing=self.packing, \
                                    max_seq_length=self.max_seq_length, \
                                    dataset_text_field="text")
        except BaseException as err:
            print('='*80)
            print("ERROR with Finetune Loader with shard data in steps!!!\n")
            print('='*80)
            print("\n")
            raise RuntimeError(err)

        self.total_training_time = time.time() - start_time
        self._log_time('Trainer Training time', finetuner.training_time, log_file=self.shattered_logfile)
        self._log_time('Total training time', self.total_training_time, log_file=self.shattered_logfile)
        #del finetuner
        return 0
    
    def finetune_shard(self, size_per_shard=1024):
        start_time = time.time()
        finetuner = LLMFinetuner(self.model, self.tokenizer, self.batch_size, log_file=self.logfile)
        if len(self.formatted_dataset)<=2:
            if len(self.formatted_dataset)==1:
                self.formatted_dataset = self.formatted_dataset["train"]
            else:
                self.formatted_dataset = concatenate_datasets([self.formatted_dataset["train"], self.formatted_dataset["test"]])
        num_shards = (len(self.formatted_dataset)+size_per_shard-1) // size_per_shard
        print(f"===================Shard dataset {self.dataset_name.split('/')[-1]} into {num_shards} parts==================")
        try:
            finetuner.tune_shard(formatted_dataset=self.formatted_dataset, \
                                peft_config=self.peft_config, \
                                training_arguments=self.training_arguments, \
                                packing=self.packing, \
                                max_seq_length=self.max_seq_length, \
                                dataset_text_field="text", \
                                num_shards=num_shards)
        except BaseException as err:
            print('='*80)
            print("ERROR with Finetune Loader with shard data!!!\n")
            print('='*80)
            print("\n")
            raise RuntimeError(err)

        self.total_training_time = time.time() - start_time
        self._log_time('Trainer Training time', finetuner.training_time, log_file=self.shattered_logfile)
        self._log_time('Total training time', self.total_training_time, log_file=self.shattered_logfile)
        #del finetuner
        return 0
    
    def finetune_iterable(self):
        
        return
          
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
    
    print(__name__+" is running!!! with batch size of "+str(batch_size)+".\n")      
    try:
        ft_singleGPU = FinetuneLoader(model_name, dataset_name, access_token, batch_size)
        ft_singleGPU.load_model()
        ft_singleGPU.load_dataset()
        #ft_singleGPU.finetune_all()
        exit_code = ft_singleGPU.finetune_shard()
        if exit_code == 0:
            print("Training Successful!!!")
            sys.exit(0)
        else:
            print("Training Failed!!!")
            sys.exit(1)
    except BaseException as err:
        print('='*80)
        print("ERROR with Main Process!!!\n")
        print('='*80)
        print("\n")
        sys.exit(1)
    sys.exit(0)