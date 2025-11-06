import json
from datasets import Dataset, DatasetDict
import random
import pandas as pd

class Jokedata:
    def __init__(self, tokenizer, path_cn, path_en, path_sp):
        self.tokenizer = tokenizer
        self.path_cn = path_cn
        self.path_en = path_en
        self.path_sp = path_sp

    def prepare_dataset(self):
        """准备训练数据集"""
        df_cn = pd.read_csv(self.path_cn, sep='\t')
        df_en = pd.read_csv(self.path_en, sep='\t')
        df_sp = pd.read_csv(self.path_sp, sep='\t')
        
        all_samples = []
        for _, joke_data in df_cn.iterrows():
            all_samples.append({
                "instruction": "请用给定的两个中文词语创作一个搞笑笑话，要求包含至少一种幽默技巧（如谐音梗、反转结局、夸张比喻或意想不到的关联），语言生动有趣，让人会心一笑：",
                "input": f"{joke_data["word1"]}, {joke_data["word2"]}",
                "output": joke_data["headline"]
            })
        for _, joke_data in df_en.iterrows():
            all_samples.append({
                "instruction": f"Create a funny English joke using the two given words. Use humor techniques like puns, unexpected twists, or silly exaggeration. Make it light-hearted and easy to laugh at:",
                "input": f"{joke_data["word1"]}, {joke_data["word2"]}",
                "output": joke_data["headline"]
            })
        for _, joke_data in df_sp.iterrows():
            all_samples.append({
                "instruction": "Crea un chiste español gracioso con las dos palabras dadas. Usa técnicas como juegos de palabras, giros inesperados o exageraciones divertidas para que cause risa:",
                "input": f"{joke_data["word1"]}, {joke_data["word2"]}",
                "output": joke_data["headline"]
            }) 
        
        # 转换为Hugging Face数据集格式
        dataset = Dataset.from_list(all_samples)           
        return dataset
    
    def format_qwen_template(self, dataset):
        def format_conversation(example):
            MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
            input_ids, attention_mask, labels = [], [], []
            instruction = self.tokenizer(f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
            response = self.tokenizer(f"{example['output']}", add_special_tokens=False)
            input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
            attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
            labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
            if len(input_ids) > MAX_LENGTH:  # 做一个截断
                input_ids = input_ids[:MAX_LENGTH]
                attention_mask = attention_mask[:MAX_LENGTH]
                labels = labels[:MAX_LENGTH]
            # elif len(input_ids) < MAX_LENGTH:
            #     # 填充到最大长度
            #     pad_length = MAX_LENGTH - len(input_ids)
            #     input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            #     attention_mask = attention_mask + [0] * pad_length
            #     labels = labels + [-100] * pad_length
        
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        formatted_dataset = dataset.map(
            format_conversation,
            remove_columns=dataset.column_names
        )
        # print(self.tokenizer.decode(formatted_dataset[0]['input_ids']))
        # print(self.tokenizer.decode(list(filter(lambda x: x != -100, formatted_dataset[0]["labels"]))))
        # raise KeyboardInterrupt()
        return formatted_dataset

    def get_dataset(self, train_size=0.9):
        origin_dataset = self.prepare_dataset()
        formatted_dataset = self.format_qwen_template(origin_dataset)
        dataset = formatted_dataset.train_test_split(test_size=1-train_size, seed=42)  
        return dataset