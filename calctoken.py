from transformers import AutoTokenizer
from datasets import load_dataset
model_name = "NousResearch/Llama-2-7b-chat-hf"

def display_dataset_info_traintest(dataset, tokenized_dataset):
    total_tokens = 0
    for attr in dataset.keys():
        print(f"Dataset {attr} has {len(dataset[attr])} examples.")
        curr_tokens = sum([len(tokens["input_ids"]) for tokens in tokenized_dataset[attr]])
        print(f"Dataset {attr} has {curr_tokens} tokens.")
        total_tokens += curr_tokens

    print(f"Total number of tokens in the dataset: {total_tokens}")
    return total_tokens

dataset_name = "BelleGroup/train_3.5M_CN"
dataset = load_dataset(dataset_name, 'all')

def custom_formatting(example):
    output_text = ""
    for round in example['conversations']:
        output_text += f'''### From: {round['from']}\n 
                    ### Text: {round['value']}'''
    return {"text":output_text}

# 加载预训练的tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 应用自定义的格式化函数
formatted_dataset = dataset.map(custom_formatting)

# 然后使用tokenizer对处理过的文本进行tokenize处理
tokenized_dataset = formatted_dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)

display_dataset_info_traintest(dataset, tokenized_dataset)