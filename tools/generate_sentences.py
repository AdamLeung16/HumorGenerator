import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from tqdm import tqdm

# ... existing model loading code ...

mode_path = '/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct/'
# lora_path = '/root/nlp_project/output/qwen_joke_lora_1031/checkpoint-7500'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
# model = PeftModel.from_pretrained(model, model_id=lora_path)
model = model.to('cuda')

# 读取测试数据
test_df = pd.read_csv('../output/5T2_output/gt.tsv',sep="\t")

# 准备存储结果的列表
results = []

# 使用tqdm添加进度条
for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="生成"):
    word1 = row['word1']
    word2 = row['word2']
    item_id = str(row['id'])
    
    # 根据id前两位确定语言
    language_code = row['language']
    
    # 根据语言设置不同的提示词
    if language_code == 'cn':  # 中文
        prompt = f"{word1},{word2}"
        system_prompt = "请用给定的两个中文词语写一句话：",
        # system_prompt = "请用给定的两个中文词语创作一个搞笑笑话，要求包含至少一种幽默技巧（如谐音梗、反转结局、夸张比喻或意想不到的关联），语言生动有趣，让人会心一笑：",
    elif language_code == 'en':  # 英文或其他语言
        prompt = f"{word1},{word2}"
        system_prompt = "Please write a sentence using the two given English words:"
    else:
        prompt = f"{word1},{word2}"
        system_prompt = "Escribe una oración utilizando las dos palabras en español dadas:"
    # 生成输入
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": f"{system_prompt}{prompt}"}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           ).to('cuda')
    
    # 生成输出
    gen_kwargs = {"max_length": 100, 
    "do_sample": True, 
    "top_k": 30, "top_p": 0.9, "repetition_penalty":1, "temperature":0.7}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 保存结果
    results.append({
        'id': row['id'],
        'word1': word1,
        'word2': word2,
        'language': language_code,
        'headline': headline
    })

# 转换为DataFrame并保存为TSV文件
output_df = pd.DataFrame(results)
output_df.to_csv('../output/5T2_output/confuse.tsv', sep='\t', index=False)

    