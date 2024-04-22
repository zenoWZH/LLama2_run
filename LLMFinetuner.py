# LLMFinetuner.py
import time
#from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
#from datasets import load_dataset
import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from DatasetFormatter import DatasetFormatter
class LLMFinetuner:
    def __init__(self, model_name, dataset_name, access_token, batch_size=4, **training_args):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        # Output directory where the model predictions and checkpoints will be stored
        self.output_dir = "./results/"
        ################################################################################
        # QLoRA parameters
        ################################################################################

        # LoRA attention dimension
        lora_r = 64

        # Alpha parameter for LoRA scaling
        lora_alpha = 16

        # Dropout probability for LoRA layers
        lora_dropout = 0.1

        ################################################################################
        # bitsandbytes parameters
        ################################################################################

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
        ################################################################################

        # Number of training epochs
        num_train_epochs = 1

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
        group_by_length = True

        # Save checkpoint every X updates steps
        save_steps = 0

        # Log every X updates steps
        logging_steps = 100

        ################################################################################
        # SFT parameters
        ################################################################################

        # Maximum sequence length to use
        self.max_seq_length = 4096

        # Pack multiple short examples in the same input sequence to increase efficiency
        self.packing = False

        # Load the entire model on the GPU 0
        device_map = {"": 0}
        
        ################################################################################
        # Load the dataset
        ################################################################################
        # Load dataset (you can process it here)
                
        self.dataset_loader = DatasetFormatter(self.dataset_name)
        self.dataset_loader.load_dataset()
        
        start_time = time.time()
        formatted_dataset = self.dataset_loader.format_dataset()
        print(formatted_dataset)
        if formatted_dataset is not None and not ('train' in formatted_dataset.column_names):
            # 数据集通常包含训练和测试集，但如果需要自定义比例，可以使用以下方法
            train_test_split = formatted_dataset.train_test_split(test_size=0.1)

            # 创建一个新的数据集字典，包括训练集和测试集
            self.split_dataset = DatasetDict({
                'train': train_test_split['train'],
                'test': train_test_split['test']
            })
        elif not('test' in formatted_dataset.column_names):
            train_test_split = formatted_dataset['train'].train_test_split(test_size=0.1)
            self.split_dataset = DatasetDict({
                'train': train_test_split['train'],
                'test': train_test_split['test']
            })
        else:
            self.split_dataset = formatted_dataset
            
        
        print(train_test_split)
        print(self.split_dataset.keys())
        # Load LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=access_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
        #self.tokenized_dataset = self.formatted_dataset.map(lambda examples: self.tokenizer(examples["text"]), batched=True)
        self.data_loading_time = time.time() - start_time
        self._log_time('Dataset preparing time', self.data_loading_time)
        ################################################################################
        # Load the model
        ################################################################################
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
            logging_dir=self.output_dir+"logs",
        )
        
    def _log_time(self, prefix, seconds):
        # Calculate time format
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        time_str = "{:0>1} days, {:0>2} hours, {:0>2} minutes, {:0>2} seconds".format(int(days), int(hours), int(minutes), int(secs))
        message = f'{prefix}: {time_str}\n'
        
        # Use temporary variables for filename
        model_short_name = self.model_name.split('/')[-1]
        dataset_short_name = self.dataset_name.split('/')[-1]
        filename = f"{model_short_name}_{dataset_short_name}_{self.batch_size}timing_log.txt"
        
        print(message.strip())
        with open(self.output_dir+filename, "w") as file:
            file.write(message)

    def train(self):
        start_time = time.time()
        
        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=self.model,
            peft_config=self.peft_config,
            train_dataset=self.split_dataset['train'],
            eval_dataset=self.split_dataset['test'],
            args=self.training_arguments,
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            packing=self.packing,
            dataset_text_field='text',
        )
        trainer.train()
        self.training_time = time.time() - start_time
        self._log_time('Training time', self.training_time)
