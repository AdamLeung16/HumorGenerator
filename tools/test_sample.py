import pandas as pd
import os
import argparse

def extract_combined_samples(seed):
    # 文件路径
    cn_file = "../data/cn_keywords.tsv"
    en_file = "../data/en_keywords.tsv"
    sp_file = "../data/sp_keywords.tsv"
    
    # 输出文件路径
    output_file = "../output/5T2_output/gt.tsv"
    
    try:
        # 读取各语言文件并抽取200条样本
        cn_df = pd.read_csv(cn_file, sep='\t')
        cn_sample = cn_df.sample(n=200, random_state=seed)
        cn_sample['language'] = 'cn'  # 添加语言标识
        
        en_df = pd.read_csv(en_file, sep='\t')
        en_sample = en_df.sample(n=200, random_state=seed)
        en_sample['language'] = 'en'
        
        sp_df = pd.read_csv(sp_file, sep='\t')
        sp_sample = sp_df.sample(n=200, random_state=seed)
        sp_sample['language'] = 'sp'
        
        # 合并所有样本
        combined_sample = pd.concat([cn_sample, en_sample, sp_sample], ignore_index=True)
        
        # 保存合并后的文件
        combined_sample.to_csv(output_file, sep='\t', index=False)
        print(f"合并样本已保存: {output_file}, 共{len(combined_sample)}条记录")
        print(f"各语言分布: 中文{len(cn_sample)}条, 英文{len(en_sample)}条, 西班牙文{len(sp_sample)}条")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    seed = args.seed
    extract_combined_samples(seed)