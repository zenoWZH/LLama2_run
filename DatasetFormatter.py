from datasets import load_dataset, concatenate_datasets
class DatasetFormatter:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        return

    def mmlu_formatting(self, example):
        output_text = f'''### Question: {example['question']}\n 
                        ### Choices: {example['choices']}\n 
                        ### Answer: {example['answer']}'''
        return {"text":output_text}
    def dolly_formatting(self, example):
        output_text = f'''### Instruction: {example['instruction']}\n 
                        ### Context: {example['context']}\n 
                        ### Answer: {example['response']}'''
        return {"text":output_text}
    def instruct_formatting(self, example):
        output_text = f'''### Instruction: {example['instruction']}\n 
                        ### Answer: {example['output']}'''
        return {"text":output_text}
    def conversation_formatting(self, example):
        output_text = ""
        for round in example['conversations']:
            output_text += f'''### From: {round['from']}\n 
                        ### Text: {round['value']}'''
        return {"text":output_text}
    
    def display_dataset_info_traintest(self, dataset, tokenized_dataset):
        total_tokens = 0
        for attr in dataset.keys():
            print(f"Dataset {attr} has {len(dataset[attr])} examples.")
            curr_tokens = sum([len(tokens["input_ids"]) for tokens in tokenized_dataset[attr]])
            print(f"Dataset {attr} has {curr_tokens} tokens.")
            total_tokens += curr_tokens

        print(f"Total number of tokens in the dataset: {total_tokens}")
        return total_tokens
    
    def display_dataset_info(tokenized_dataset):
        total_tokens = 0
        attr = "text"
        #for attr in tokenized_dataset.features.keys():
        print(f"Dataset {attr} has {len(tokenized_dataset[attr])} examples.")
        curr_tokens = sum([len(tokens["input_ids"]) for tokens in tokenized_dataset])
        print(f"Dataset {attr} has {curr_tokens} tokens.")
        total_tokens += curr_tokens

        print(f"Total number of tokens in the dataset: {total_tokens}")
    
    def load_dataset(self):
        if self.dataset_name == "cais/mmlu":
            self.dataset = load_dataset("cais/mmlu", name = "all",  split="auxiliary_train")
        elif self.dataset_name == "databricks/databricks-dolly-15k":
            self.dataset = load_dataset("databricks/databricks-dolly-15k", 'all')
        elif self.dataset_name == "BelleGroup/train_1M_CN":
            self.dataset = load_dataset("BelleGroup/train_1M_CN", 'all')
        elif self.dataset_name == "BelleGroup/train_0.5M_CN":
            self.dataset = load_dataset("BelleGroup/train_0.5M_CN",  'all')
        elif self.dataset_name == "BelleGroup/train_3.5M_CN":
            self.dataset = load_dataset("BelleGroup/train_3.5M_CN",  'all')
        elif self.dataset_name == "m-a-p/COIG-CQIA":
            self.dataset = load_dataset("m-a-p/COIG-CQIA", 'chinese_traditional', split="train")
            loadlist = ['coig_pc', 'exam', 'finance', 'douban', 'human_value', 'logi_qa', 'ruozhiba', 'segmentfault', 'wiki', 'wikihow', 'xhs', 'zhihu']
            for option in loadlist:
                self.dataset = concatenate_datasets([self.dataset, load_dataset("m-a-p/COIG-CQIA", option, split="train")])
                
        else:
            print("Dataset name not recognized.")
            try:
                self.dataset = load_dataset(self.dataset_name, split="train")
                self.dataset = concatenate_datasets([self.dataset, load_dataset(self.dataset_name, split="test")])
            except BaseException as err:
                print(err)
                return None
    
    def format_dataset(self):
        if self.dataset_name == "cais/mmlu":
            return self.dataset.map(self.mmlu_formatting)
        elif self.dataset_name == "databricks/databricks-dolly-15k":
            return self.dataset.map(self.dolly_formatting)
        elif self.dataset_name == "BelleGroup/train_1M_CN" \
            or self.dataset_name == "BelleGroup/train_0.5M_CN" \
                or self.dataset_name == "m-a-p/COIG-CQIA":
            return self.dataset.map(self.instruct_formatting)
        elif self.dataset_name == "BelleGroup/train_3.5M_CN":
            return self.dataset.map(self.conversation_formatting)
        else:
            print("Dataset name not recognized.")
            return self.dataset.with_format("torch")