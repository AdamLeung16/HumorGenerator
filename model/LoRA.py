import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from jokedataset import Jokedata

class JokeTrainer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def setup_model(self):
        """初始化模型和tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.enable_input_require_grads()
        
    def setup_lora(self):
        """配置LoRA"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def train(self, dataset, output_dir):
        """训练模型"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=3,

            learning_rate=1e-5,
            weight_decay=0.1,
            warmup_ratio=0.1,
            max_grad_norm=1.0,  # 添加梯度裁剪
            # learning_rate=5e-6,
            #fp16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},  # 抑制警告并提升稳定性
            # warmup_steps=100,
            # logging_steps=10,
            # eval_strategy="steps",
            # eval_steps=100,
            # save_steps=100,
            # metric_for_best_model="eval_loss",  # 或您自定义的评估指标
            # greater_is_better=False,      # 如果使用loss，越小越好
            # save_total_limit=1,
            # load_best_model_at_end=True,

            # 日志配置
            logging_strategy="steps",
            logging_steps=20,

            eval_strategy="steps",      # 每个epoch结束后评估
            eval_steps=500,                   # 如果使用steps策略
            save_strategy="steps",            # 保存策略与评估策略一致
            load_best_model_at_end=True,      # 关键：训练结束时加载最佳模型
            metric_for_best_model="eval_loss", # 根据验证集loss选择最佳模型
            greater_is_better=False,          # loss越小越好
            save_total_limit=3,               # 只保留3个检查点（包含最佳模型）
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            # tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True),
        )
        
        print("开始训练...")
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        print("训练完成!")

# 使用示例
if __name__ == "__main__":
    # 初始化训练器
    trainer = JokeTrainer(model_name="/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct/")
    trainer.setup_model()
    trainer.setup_lora()
    
    # 准备数据（这里需要替换为您的实际数据）
    path_cn = "../data/cn_keywords.tsv"
    path_en = "../data/en_keywords.tsv"
    path_sp = "../data/sp_keywords.tsv"
    Data = Jokedata(trainer.tokenizer,path_cn,path_en,path_sp)
    dataset = Data.get_dataset(train_size=0.9)
    
    # 开始训练
    trainer.train(dataset, output_dir="../checkpoints")