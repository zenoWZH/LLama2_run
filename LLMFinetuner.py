# LLMFinetuner.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from tqdm import tqdm
import shutil
import os

class LLMFinetuner:
    def __init__(self, model_name, dataset_name, **training_args):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.dataset = load_dataset(dataset_name)
        self.training_args = training_args

    def train(self):
        training_args = TrainingArguments(output_dir="./results", **self.training_args)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"]
        )
        for epoch in tqdm(range(int(self.training_args.get("num_train_epochs", 1))), desc="Training Epochs"):
            trainer.train()

    def clear_cache(self):
        cache_dir = os.path.join(os.getcwd(), '.cache', 'huggingface')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
