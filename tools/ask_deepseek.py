import pandas as pd
import os
import requests
import json
import time
from typing import List, Dict
from tqdm import tqdm

def extract_headlines_from_tsv(file_path: str) -> List[str]:
    """从TSV文件中提取headline列"""
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    return df['headline'].tolist()

def create_prompts(headlines_by_file: Dict[str, List[str]]) -> List[str]:
    """创建prompt，将同一行的headline组合在一起"""
    num_lines = len(list(headlines_by_file.values())[0])
    prompts = []
    
    for line_idx in range(num_lines):
        prompt_parts = []
        for file_id, headlines in headlines_by_file.items():
            if line_idx < len(headlines):
                headline = str(headlines[line_idx]).strip()
                if headline and headline != 'nan':  # 过滤空值和NaN
                    prompt_parts.append(f"文件{file_id}: {headline}")
        
        if prompt_parts:  # 确保有内容
            prompt = "请从以下幽默笑话中选择最幽默的2个，只输出文件编号（1-5），编号之间用英文逗号隔开，不需要其他任何解释：\n\n" + "\n\n".join(prompt_parts)
            prompts.append(prompt)
    
    return prompts

def call_deepseek_api(prompt: str, api_key: str) -> str:
    """调用DeepSeek API获取响应"""
    url = "https://api.deepseek.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            processing_time = end_time - start_time
            # print(f"API调用成功，耗时: {processing_time:.2f}秒")
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"API调用失败: {response.status_code}"
    except requests.exceptions.Timeout:
        return "API调用超时"
    except Exception as e:
        return f"API调用异常: {str(e)}"

def main():
    # TSV文件目录
    directory = "../output/5T2_output"
    
    # 文件映射
    file_mapping = {
        "1": "gt.tsv",
        "2": "confuse.tsv",
        "3": "qwen.tsv", 
        "4": "glm.tsv",
        "5": "ours.tsv"
    }
    vote_cnt = {"1":0,"2":0,"3":0,"4":0,"5":0}
    
    # 请在此处设置您的API密钥
    API_KEY = "sk-cee324a0ab9a43a6bfc7eb7bdb273cd1"  # 替换为您的实际API密钥
    
    print("开始处理TSV文件...")
    
    # 从所有文件提取headline
    headlines_by_file = {}
    for file_id, file_name in file_mapping.items():
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            try:
                headlines = extract_headlines_from_tsv(file_path)
                headlines_by_file[file_id] = headlines
                print(f"✓ 从 {file_name} (文件{file_id}) 加载了 {len(headlines)} 个headline")
            except Exception as e:
                print(f"✗ 读取文件 {file_name} 失败: {e}")
        else:
            print(f"✗ 文件不存在: {file_path}")
    
    # 创建prompts
    prompts = create_prompts(headlines_by_file)
    print(f"\n创建了 {len(prompts)} 个prompt")
    
    # 询问用户要处理多少条记录
    total_prompts = len(prompts)
    # print(f"\n共有 {total_prompts} 条记录需要处理")
    
    # try:
    #     num_to_process = int(input(f"请输入要处理的记录数量 (1-{total_prompts}): "))
    #     num_to_process = min(max(1, num_to_process), total_prompts)
    # except:
    #     num_to_process = min(10, total_prompts)  # 默认处理10条
    #     print(f"使用默认值: {num_to_process} 条")
    
    # 处理prompt并获取响应
    results = []
    # print(f"\n开始处理前 {num_to_process} 条记录...")
    
    start_total_time = time.time()
    
    for i in tqdm(range(total_prompts)):
        response = ""
        prompt = prompts[i]
        print(f"\n处理第 {i+1}/{total_prompts} 行...")
        while(len(response)!=3):
            response = call_deepseek_api(prompt, API_KEY)
            print(f"响应: {response}")
        results.append({
            "line_number": i + 1,
            "response": response
        })
        vote = response.split(",")
        for v in vote:
            vote_cnt[v] += 1
        # 添加延迟避免API限制
        time.sleep(1)
    
    end_total_time = time.time()
    total_processing_time = end_total_time - start_total_time
    
    print(f"\n支持率统计:")
    for key, value in vote_cnt.items():
        acc = value / total_prompts
        name = file_mapping[key]
        print(f"{name}: {acc:.2f}")
        results.append({
                name: acc
            })
    # 保存结果
    with open("../output/5T2_output/humor_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 处理完成!")
    print(f"✓ 总耗时: {total_processing_time:.2f}秒")
    print(f"✓ 平均每条记录: {total_processing_time/total_prompts:.2f}秒")
    print(f"✓ 结果已保存到 humor_results.json")

if __name__ == "__main__":
    main()