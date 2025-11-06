from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = '/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct/'
lora_path = '../checkpoints/checkpoint-7500' # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)
model = model.to('cuda')

prompt = "移动,冰箱"
inputs = tokenizer.apply_chat_template([{"role": "user", 
                                        "content": f"请用给定的两个中文词语创作一个搞笑笑话，语言生动有趣，让人会心一笑:{prompt}"
                                        # "content": f"请用给定的两个中文词语创作一个搞笑笑话，要求包含至少一种幽默技巧（如谐音梗、反转结局、夸张比喻或意想不到的关联），语言生动有趣，让人会心一笑：{prompt}"
                                        }],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to('cuda')


gen_kwargs = {"max_length": 200, "do_sample": True, "top_k": 50, "top_p": 0.95, "repetition_penalty":1.3, "temperature":0.8}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))