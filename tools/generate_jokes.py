import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="Model name")
args = parser.parse_args()

# ... existing model loading code ...
model_path_dict = {"qwen":"/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct",
                "glm":"/root/autodl-tmp/glm/ZhipuAI/glm-4-9b",
                "ours":"/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"}
# 配置模型路径 - 请替换为您的实际路径
model_name = args.model
model_path = model_path_dict[model_name]

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
if model_name=="ours": 
    lora_path = '../checkpoints/checkpoint-7500'
    model = PeftModel.from_pretrained(model, model_id=lora_path)
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
        system_prompt = "请用给定的两个中文词语创作一个搞笑笑话，语言生动有趣，让人会心一笑：",
        # system_prompt = "请用给定的两个中文词语创作一个搞笑笑话，要求包含至少一种幽默技巧（如谐音梗、反转结局、夸张比喻或意想不到的关联），语言生动有趣，让人会心一笑：",
    elif language_code == 'en':  # 英文或其他语言
        prompt = f"{word1},{word2}"
        system_prompt = "Create a funny English joke using the two given words. Make it light-hearted and easy to laugh at:"
    else:
        prompt = f"{word1},{word2}"
        system_prompt = "Crea un chiste español gracioso con las dos palabras dadas:"
    # 生成输入
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": f"{system_prompt}{prompt}"}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           ).to('cuda')
    
    # 生成输出
    gen_kwargs = {"max_length": 200, 
    "do_sample": True, 
    "top_k": 50, "top_p": 0.95, "repetition_penalty":1.3, "temperature":0.8}
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
output_file = f"../output/5T2_output/{model_name}.tsv"
output_df.to_csv(output_file, sep='\t', index=False)

    