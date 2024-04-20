# main.py
from LLMFinetuner import LLMFinetuner
from ConfigReader import ConfigReader
import sys
from tqdm import tqdm

if __name__ == "__main__":
    if len(sys.argv) == 3:
        model_name = sys.argv[1]
        dataset_name = sys.argv[2]
        finetuner = LLMFinetuner(model_name, dataset_name, per_device_train_batch_size=8, num_train_epochs=3)
        finetuner.train()
    else:
        model_reader = ConfigReader("models.txt")
        dataset_reader = ConfigReader("datasets.txt")
        models = model_reader.read_lines_without_comments()
        datasets = dataset_reader.read_lines_without_comments()
        for model in tqdm(models, desc="Models"):
            for dataset in tqdm(datasets, desc="Datasets", leave=False):
                finetuner = LLMFinetuner(model, dataset, per_device_train_batch_size=8, num_train_epochs=3)
                finetuner.train()
