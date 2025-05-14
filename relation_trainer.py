#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 核心功能板块

"""
Relation Trainer for Knowledge Graph

用于训练和预测实体间关系的模块
"""

import os
import numpy as np
import torch
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, AutoTokenizer
from embedding_manager import EmbeddingManager
from pathlib import Path
from tqdm import tqdm

class RelationDataset(Dataset):
    """关系数据集类"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除批次维度
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(self.labels[idx])
        
        return inputs

class RelationTrainer:
    """关系训练器类"""
    
    def __init__(self, config_file=None, model_dir="./models/relation_model"):
        """初始化关系训练器
        
        Args:
            config_file: 配置文件路径
            model_dir: 模型保存目录
        """
        # 初始化嵌入管理器
        self.embedding_manager = EmbeddingManager(config_file)
        
        # 创建模型目录
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        self.model_dir = model_dir
        
        # 关系类型映射
        self.relation_types = [
            "positive_regulation",   # 正向调控，如激活、促进
            "negative_regulation",   # 负向调控，如抑制、减少
            "regulation",            # 一般调控，无明确方向
            "association",           # 相关性，无因果关系
            "conversion",            # 转化/转变关系
            "DEFAULT"                # 默认关系
        ]
        
        self.label2id = {label: i for i, label in enumerate(self.relation_types)}
        self.id2label = {i: label for i, label in enumerate(self.relation_types)}
        
        # 初始化模型
        self.initialize_model()
        
    def initialize_model(self):
        """初始化关系分类模型"""
        try:
            # 优先使用本地模型路径
            model_path = None
            
            # 首先尝试从embedding_manager获取本地模型路径
            local_model_path = getattr(self.embedding_manager, 'local_model_path', None)
            if local_model_path and os.path.exists(local_model_path):
                model_path = local_model_path
                print(f"使用embedding_manager中的本地模型路径: {model_path}")
            
            # 如果没有找到有效的本地路径，尝试固定路径
            if not model_path or not os.path.isdir(model_path):
                # 尝试直接使用绝对路径
                abs_path = "D:/article_tho/copd_inhalation_tcm/liter_ana/models/tinybert"
                if os.path.isdir(abs_path):
                    model_path = abs_path
                    print(f"使用绝对模型路径: {model_path}")
                else:
                    # 尝试相对路径
                    rel_path = "./models/tinybert"
                    if os.path.isdir(rel_path):
                        model_path = rel_path
                        print(f"使用相对模型路径: {model_path}")
            
            # 如果仍然没有找到模型路径，使用模型名称
            if not model_path or not os.path.isdir(model_path):
                model_name = getattr(self.embedding_manager, 'model_name', None)
                model_path = model_name
                print(f"警告: 未找到本地模型目录，尝试使用模型名称: {model_path}")
            
            # 如果路径存在且是目录，则使用本地模型
            if model_path and os.path.isdir(model_path):
                print(f"使用本地模型: {model_path}")
                
                # 检查必要的文件
                required_files = ["config.json", "pytorch_model.bin", "vocab.txt"]
                missing_files = []
                for file in required_files:
                    if not os.path.exists(os.path.join(model_path, file)):
                        missing_files.append(file)
                
                if missing_files:
                    print(f"警告: 本地模型目录缺少必要文件: {', '.join(missing_files)}")
                    raise ValueError(f"模型目录不完整，请确保包含所有必要文件: {', '.join(required_files)}")
                
                # 从本地加载tokenizer
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
                
                # 从本地创建配置，并指定为bert类型
                config = BertConfig.from_pretrained(model_path)
                if not hasattr(config, 'model_type') or not config.model_type:
                    print("将model_type设置为'bert'")
                    config.model_type = "bert"
                    # 保存修复后的配置
                    config.save_pretrained(model_path)
                    print(f"已将修复后的配置保存到: {os.path.join(model_path, 'config.json')}")
                
                # 设置分类任务所需的配置
                config.num_labels = len(self.relation_types)
                config.id2label = self.id2label
                config.label2id = self.label2id
                
                # 使用BERT分类器加载模型
                self.model = BertForSequenceClassification.from_pretrained(
                    model_path,
                    config=config
                )
                print("√ 模型初始化成功")
            else:
                # 如果没有有效的模型路径
                print(f"错误: 无法找到有效的模型路径或目录")
                raise ValueError("请确保本地模型目录存在并包含必要文件，或使用test_relation_trainer.py下载并修复TinyBERT模型")
            
            # 检查是否存在预训练的关系模型
            relation_model_path = os.path.join(self.model_dir, "pytorch_model.bin")
            if os.path.exists(relation_model_path):
                print(f"加载已训练的关系模型: {relation_model_path}")
                self.model.load_state_dict(torch.load(relation_model_path))
                
            # 移至适当设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            print(f"关系分类模型初始化完成，使用设备: {self.device}")
            self.model_initialized = True
            
        except Exception as e:
            logging.error(f"初始化关系模型时出错: {str(e)}")
            print(f"无法初始化关系模型: {str(e)}")
            self.model_initialized = False
    
    def prepare_training_data(self, data_file=None, custom_examples=None):
        """准备训练数据
        
        Args:
            data_file: 包含训练数据的CSV文件
            custom_examples: 自定义的训练示例列表
            
        Returns:
            bool: 数据准备是否成功
        """
        try:
            self.train_texts = []
            self.train_labels = []
            
            # 如果提供了数据文件，从文件加载
            if data_file and os.path.exists(data_file):
                print(f"从文件加载训练数据: {data_file}")
                df = pd.read_csv(data_file)
                for _, row in df.iterrows():
                    text = f"{row['source_entity']} [SEP] {row['target_entity']}"
                    label = self.label2id.get(row['relation_type'], self.label2id['DEFAULT'])
                    self.train_texts.append(text)
                    self.train_labels.append(label)
            
            # 如果提供了自定义示例，添加到训练数据
            if custom_examples:
                print(f"添加 {len(custom_examples)} 个自定义训练示例")
                for source, relation, target in custom_examples:
                    text = f"{source} [SEP] {target}"
                    label = self.label2id.get(relation, self.label2id['DEFAULT'])
                    self.train_texts.append(text)
                    self.train_labels.append(label)
            
            # 如果没有数据，添加一些基本示例
            if not self.train_texts:
                print("没有提供训练数据，添加基本示例...")
                basic_examples = [
                    ("AMPK", "positive_regulation", "autophagy"),
                    ("mTOR", "negative_regulation", "autophagy"),
                    ("FOXP3", "positive_regulation", "Treg"),
                    ("IL-2", "positive_regulation", "T cell"),
                    ("inflammation", "association", "autoimmune"),
                    ("T cell", "conversion", "Treg"),
                    ("LC3", "association", "autophagosome"),
                    ("rapamycin", "negative_regulation", "mTOR"),
                    ("FOXP3", "regulation", "CD25"),
                    ("CTLA-4", "negative_regulation", "T cell activation")
                ]
                
                for source, relation, target in basic_examples:
                    text = f"{source} [SEP] {target}"
                    label = self.label2id.get(relation, self.label2id['DEFAULT'])
                    self.train_texts.append(text)
                    self.train_labels.append(label)
            
            # 拆分训练和验证数据
            self.train_texts, self.val_texts, self.train_labels, self.val_labels = train_test_split(
                self.train_texts, self.train_labels, test_size=0.2, random_state=42
            )
            
            print(f"准备完成，共 {len(self.train_texts)} 条训练数据，{len(self.val_texts)} 条验证数据")
            return True
            
        except Exception as e:
            logging.error(f"准备训练数据时出错: {str(e)}")
            print(f"无法准备训练数据: {str(e)}")
            return False
    
    def train(self, epochs=5, batch_size=16, learning_rate=5e-5):
        """训练关系分类模型
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            
        Returns:
            dict: 训练结果统计
        """
        if not self.model_initialized:
            print("模型未初始化，无法训练")
            return None
            
        if not hasattr(self, 'train_texts') or len(self.train_texts) == 0:
            print("没有训练数据，请先准备训练数据")
            return None
        
        try:
            # 更新优化器导入
            from torch.optim import AdamW
            from transformers import get_linear_schedule_with_warmup
            
            # 创建数据集和数据加载器
            train_dataset = RelationDataset(self.train_texts, self.train_labels, self.tokenizer)
            val_dataset = RelationDataset(self.val_texts, self.val_labels, self.tokenizer)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            
            # 设置优化器和学习率调度器
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            num_training_steps = len(train_dataloader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=0,
                num_training_steps=num_training_steps
            )
            
            # 训练循环
            print(f"开始训练，共 {epochs} 轮...")
            self.model.train()
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                
                total_loss = 0
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
                
                for batch in progress_bar:
                    # 将数据移至设备
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # 前向传播
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # 更新进度条
                    progress_bar.set_postfix({'loss': loss.item()})
                
                # 每轮结束后评估
                val_accuracy = self.evaluate(val_dataloader)
                print(f"Epoch {epoch+1} - Avg. Loss: {total_loss/len(train_dataloader):.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # 保存模型
            print(f"保存模型到 {self.model_dir}")
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            
            return {
                "epochs": epochs,
                "final_loss": total_loss/len(train_dataloader),
                "val_accuracy": val_accuracy
            }
            
        except Exception as e:
            logging.error(f"训练模型时出错: {str(e)}")
            print(f"训练失败: {str(e)}")
            return None
    
    def evaluate(self, dataloader=None):
        """评估模型性能
        
        Args:
            dataloader: 数据加载器，如果为None则使用验证数据
            
        Returns:
            float: 准确率
        """
        if not self.model_initialized:
            print("模型未初始化，无法评估")
            return 0.0
            
        if dataloader is None:
            if not hasattr(self, 'val_texts') or len(self.val_texts) == 0:
                print("没有验证数据，无法评估")
                return 0.0
                
            val_dataset = RelationDataset(self.val_texts, self.val_labels, self.tokenizer)
            dataloader = DataLoader(val_dataset, batch_size=16)
        
        # 评估模式
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop('labels')
                
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += len(labels)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 打印分类报告，增加错误处理
        if len(all_preds) > 0:
            try:
                # 获取实际使用的类别
                unique_preds = np.unique(all_preds)
                unique_labels = np.unique(all_labels)
                used_classes = np.unique(np.concatenate([unique_preds, unique_labels]))
                
                # 创建只包含使用类别的标签名称
                used_target_names = [self.id2label[i] for i in used_classes]
                
                print("\n分类报告:")
                report = classification_report(
                    all_labels, 
                    all_preds, 
                    target_names=used_target_names,
                    labels=list(used_classes),
                    zero_division=0
                )
                print(report)
            except Exception as e:
                print(f"生成分类报告时出错: {str(e)}")
                print(f"预测的类别: {np.unique(all_preds)}")
                print(f"真实的类别: {np.unique(all_labels)}")
        
        return correct / total if total > 0 else 0.0
    
    def predict_relation(self, source_entity, target_entity):
        """预测两个实体之间的关系
        
        Args:
            source_entity: 源实体
            target_entity: 目标实体
            
        Returns:
            str: 预测的关系类型
            float: 关系概率
        """
        if not self.model_initialized:
            print("模型未初始化，无法预测")
            return "DEFAULT", 0.0
        
        # 准备输入
        text = f"{source_entity} [SEP] {target_entity}"
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()
        
        relation_type = self.id2label[prediction]
        return relation_type, confidence
    
    def predict_batch(self, entity_pairs):
        """批量预测实体对之间的关系
        
        Args:
            entity_pairs: 实体对列表 [(source1, target1), (source2, target2), ...]
            
        Returns:
            list: 关系预测结果列表 [(relation1, confidence1), (relation2, confidence2), ...]
        """
        results = []
        for source, target in entity_pairs:
            relation, confidence = self.predict_relation(source, target)
            results.append((relation, confidence))
        return results

# 如果直接运行此文件，执行简单测试
if __name__ == "__main__":
    print("Testing Relation Trainer...")
    
    # 创建训练器
    trainer = RelationTrainer()
    
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
        ("CTLA-4", "negative_regulation", "T cell activation")
    ]
    
    # 准备训练数据
    trainer.prepare_training_data(custom_examples=examples)
    
    # 训练模型（少量epochs用于测试）
    results = trainer.train(epochs=2)
    print(f"训练结果: {results}")
    
    # 测试预测
    test_pairs = [
        ("p53", "autophagy"),
        ("Beclin1", "autophagosome"),
        ("IL-10", "Treg"),
        ("diabetes", "inflammation")
    ]
    
    print("\n预测结果:")
    predictions = trainer.predict_batch(test_pairs)
    for i, (pair, pred) in enumerate(zip(test_pairs, predictions)):
        print(f"{pair[0]} → {pair[1]}: {pred[0]} (置信度: {pred[1]:.2f})")
    
    print("\nTest completed!")
