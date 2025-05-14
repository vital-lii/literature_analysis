#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -i 指定文件路径 -e 指定训练轮数 -b 指定批次大小 -t 训练完成后进行测试 
# 主程序 

"""
训练关系模型并应用于知识图谱
"""

import os
import sys
import pandas as pd
import argparse
from relation_trainer import RelationTrainer
from knowledge_graph import KnowledgeGraphBuilder

def main():
    parser = argparse.ArgumentParser(description='训练关系模型并应用于知识图谱')
    parser.add_argument('-c', '--config', default='config/autophagy_treg_config.yaml', help='配置文件路径')
    parser.add_argument('-i', '--data', help='关系训练数据文件路径')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('-t', '--test', action='store_true', help='是否运行测试预测')
    args = parser.parse_args()
    
    # 初始化关系训练器
    trainer = RelationTrainer(config_file=args.config)
    
    # 准备一些示例数据
    examples = [
        ("AMPK", "positive_regulation", "autophagy"),
        ("mTOR", "negative_regulation", "autophagy"),
        ("FOXP3", "positive_regulation", "Treg"),
        ("IL-2", "positive_regulation", "T cell"),
        ("inflammation", "association", "autoimmune"),
        ("T cell", "conversion", "Treg"),
        ("LC3", "association", "autophagosome"),
        ("rapamycin", "negative_regulation", "mTOR"),
        ("FOXP3", "regulation", "CD25"),
        ("CTLA-4", "negative_regulation", "T cell activation"),
        
        # 添加更多生物学关系示例
        ("p62", "association", "autophagy"),
        ("ATG5", "positive_regulation", "autophagosome"),
        ("STAT3", "positive_regulation", "Treg"),
        ("TNF-α", "negative_regulation", "Treg"),
        ("IL-10", "positive_regulation", "Treg"),
        ("TGF-β", "positive_regulation", "Treg"),
        ("Beclin1", "positive_regulation", "autophagy"),
        ("chloroquine", "negative_regulation", "autophagy"),
        ("bafilomycin", "negative_regulation", "autophagy"),
        ("p53", "regulation", "autophagy")
    ]
    
    # 准备训练数据
    trainer.prepare_training_data(data_file=args.data, custom_examples=examples)
    
    # 训练模型
    print(f"开始训练关系模型，共 {args.epochs} 轮...")
    results = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    
    if results:
        print(f"训练完成，验证准确率: {results.get('val_accuracy', 0):.4f}")
    else:
        print("训练失败")
        sys.exit(1)
    
    # 如果需要，运行测试预测
    if args.test:
        test_pairs = [
            ("p53", "autophagy"),
            ("Beclin1", "autophagosome"),
            ("IL-10", "Treg"),
            ("diabetes", "inflammation"),
            ("rapamycin", "mTOR"),
            ("FOXP3", "CD4+CD25+"),
            ("IL-6", "Treg"),
            ("autophagy", "apoptosis")
        ]
        
        print("\n测试预测结果:")
        predictions = trainer.predict_batch(test_pairs)
        for i, (pair, pred) in enumerate(zip(test_pairs, predictions)):
            print(f"{pair[0]} → {pair[1]}: {pred[0]} (置信度: {pred[1]:.2f})")
    
    print("\n关系模型训练完成，可以在知识图谱构建过程中使用")

if __name__ == "__main__":
    main()
