# 从数据集中提取测试样例 seed
python ../tools/test_sample.py 1031
# 生成混淆句子
python ../tools/generate_sentences.py
# 用预训练模型生成笑话
python ../tools/generate_jokes.py qwen
python ../tools/generate_jokes.py glm
# 用LoRa微调后的模型生成笑话
python ../tools/generate_jokes.py ours
# 用deepseek作为Agent评估5T2
python ../tools/ask_deepseek.py