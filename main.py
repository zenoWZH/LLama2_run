# main.py
from LLMFinetuner import LLMFinetuner
import sys
from tqdm import tqdm

if __name__ == "__main__":
    if len(sys.argv) == 3:
        model_name = sys.argv[1]
        dataset_name = sys.argv[2]
        finetuner = LLMFinetuner(model_name, dataset_name, per_device_train_batch_size=8, num_train_epochs=3)
        finetuner.train()
    else:
        with open("models.txt", "r") as m_file, open("datasets.txt", "r") as d_file:
            models = m_file.read().splitlines()
            datasets = d_file.read().splitlines()
            for model in tqdm(models, desc="Models"):
                for dataset in tqdm(datasets, desc="Datasets", leave=False):
                    finetuner = LLMFinetuner(model, dataset, per_device_train_batch_size=8, num_train_epochs=3)
                    finetuner.train()
