from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModel

# 加载模型和tokenizer
model_path = "/root/autodl-tmp/glm"  # 替换为您的模型路径
model_dir = snapshot_download('ZhipuAI/glm-4-9b', cache_dir=model_path)

# 从下载的目录加载
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
print(f"模型已下载到: {model_dir}")