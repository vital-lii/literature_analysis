#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试关系训练器

这个脚本用于测试TinyBERT模型在关系分类任务上的表现
"""

import os
import sys
import logging
import torch
from pathlib import Path
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_tinybert_initialization():
    """测试TinyBERT模型初始化"""
    print("\n测试TinyBERT模型初始化...")
    
    
    model_path = "./models/tinybert"
    
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print("请确保已下载TinyBERT模型")
        return False
    
    try:
        required_files = ["config.json", "pytorch_model.bin", "vocab.txt"]
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                print(f"错误: 必需的文件不存在: {file_path}")
                return False
            print(f"√ 找到文件: {file}")
        
        print("\n加载tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        print("√ Tokenizer加载成功")
        
        print("\n加载配置...")
        config = BertConfig.from_pretrained(model_path)
        if not hasattr(config, 'model_type'):
            print("! 配置中没有model_type字段，添加'bert'作为model_type")
            config.model_type = 'bert'
        print(f"√ 配置加载成功: {config}")
        
        print("\n加载模型...")
        relation_types = [
            "positive_regulation", "negative_regulation", "regulation",
            "association", "conversion", "DEFAULT"
        ]
        id2label = {i: label for i, label in enumerate(relation_types)}
        label2id = {label: i for i, label in enumerate(relation_types)}
        
        config.num_labels = len(relation_types)
        config.id2label = id2label
        config.label2id = label2id
        
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            config=config
        )
        print("√ 模型加载成功")
        
        print("\n测试模型向前传播...")
        inputs = tokenizer("Test input [SEP] Second input", padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        outputs = model(**inputs)
        print(f"√ 模型输出形状: {outputs.logits.shape}")
        
        print("\n模型初始化测试通过!")
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        logging.error(f"测试TinyBERT失败: {str(e)}")
        return False

def save_fixed_config():
    """修复配置文件并保存"""
    print("\n尝试修复配置文件...")
    
    model_path = "./models/tinybert"
    config_path = os.path.join(model_path, "config.json")
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return False
    
    try:
        config = BertConfig.from_pretrained(model_path)
        
        config.model_type = 'bert'
        
        config.save_pretrained(model_path)
        print(f"√ 修复后的配置已保存到: {config_path}")
        return True
        
    except Exception as e:
        print(f"修复配置失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("TinyBERT关系训练器测试")
    print("========================")
    
    if not test_tinybert_initialization():
        if save_fixed_config():
            print("配置已修复，重新测试...")
            test_tinybert_initialization()
    
    print("\n测试完成!")

if __name__ == "__main__":
    main() 