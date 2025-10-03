from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import pandas as pd
import re

# 模型路径和数据路径
mode_path = '/root/glm-4-9b-chat'
lora_path = '/root/GLM-4/finetune_demo/output/微调来分类文献'
data = pd.read_excel('/root/文献信息_处理完成.xlsx')

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 分类范围（可以自定义）
category_range = [
    "案例类文献", "二手数据分析类文献", "方法工具类文献", "决策模拟类文献", "理论观点类文献",
    "内容分析类文献", "实验类文献", "书评", "问卷调查类文献", "新闻观点与会议资讯", "预测建模类文献",
    "综述类文献"
]

# 定义分类函数，提取分类类别
def classify_text(title, abstract):
    # 创建输入格式
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": f"以下是论文《{title}》的摘要。根据摘要，请你判断这篇文献属于以下中的哪一类：“案例类文献”，“二手数据分析类文献”，“方法工具类文献”，“决策模拟类文献”，“理论观点类文献”，“内容分析类文献”，“实验类文献”，“书评”，“问卷调查类文献”，“新闻观点与会议资讯”，“预测建模类文献”，“综述类文献”：{abstract}"}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           ).to('cuda')

    # 设置生成参数
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

    # 生成输出
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    # 输出的格式假设类似“文献类别: <category>”，调整正则表达式来提取分类结果
    for category in category_range:
        if category in result:
            return category
    
    return "未知类别"  # 如果没有匹配到分类

# 批量处理并保存分类结果
def classify_and_save(data, output_file):
    # 创建新的列来保存分类结果
    classifications = []

    # 逐条处理数据
    for i in range(len(data)):
        title = data["title"][i]
        abstract = data["abstract"][i]

        # 分类
        classification = classify_text(title, abstract)
        classifications.append(classification)

        print(f"Processing {i+1}/{len(data)}: {title} -> {classification}")  # 可选：打印进度

    # 将分类结果添加到原始数据
    data['classification'] = classifications

    # 保存到新的Excel文件
    data.to_excel(output_file, index=False)
    print(f"分类完成，结果已保存至 {output_file}")

# 执行分类并保存结果
output_path = "/root/output_文献信息_处理完成.xlsx"
classify_and_save(data, output_path)
