#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 有向量模块
"""
Embedding Manager for Knowledge Graph

This module provides vector encoding capabilities for entities and relations 
in knowledge graphs, supporting various embedding methods.
"""

import os
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import torch
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.path_manager import PathManager
import requests
import ssl
import warnings

# 警告
warnings.filterwarnings("ignore", message=".*certificate verify failed.*")

class EmbeddingManager:
    """向量编码管理器，用于生成和管理实体与关系的向量表示"""
    
    def __init__(self, config_file=None, embedding_dim=384, use_gpu=True):
        """初始化向量编码管理器
        
        Args:
            config_file: 配置文件路径
            embedding_dim: 向量维度 (默认384, 适合大多数transformer模型)
            use_gpu: 是否使用GPU加速
        """
        self.path_manager = PathManager(config_file) if config_file else None
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # 实体和关系的编码缓存
        self.entity_embeddings = {}
        self.relation_embeddings = {}
        
        # 加载配置
        self.model_name = 'all-MiniLM-L6-v2'  # 默认轻量级模型，但不好用
        self.load_config()
        
        # 初始化编码模型
        self.initialize_model()
        
    def load_config(self):
        """从配置文件加载设置"""
        try:
            if not self.path_manager or not hasattr(self.path_manager, 'config'):
                return
                
            config = self.path_manager.config
            if 'embedding' in config:
                emb_config = config['embedding']
                
                
                if 'model' in emb_config:
                    self.model_name = emb_config['model']
                    
                # 加载维度
                if 'dimension' in emb_config:
                    self.embedding_dim = emb_config['dimension']
                
                
        except Exception as e:
            logging.warning(f"加载向量编码配置时出错: {str(e)}")
    
    def initialize_model(self):
        """初始化编码模型"""
        try:
            print(f"初始化向量编码模型: {self.model_name}")
            
            # 尝试使用离线模型（配置了本地模型路径）
            local_model_path = None
            if self.path_manager and hasattr(self.path_manager, 'config'):
                if 'embedding' in self.path_manager.config:
                    local_model_path = self.path_manager.config['embedding'].get('local_model_path', None)
            
            if local_model_path and os.path.exists(local_model_path):
                print(f"使用本地模型: {local_model_path}")
                self.model = SentenceTransformer(local_model_path, device=self.device)
                print(f"本地模型加载成功，使用设备: {self.device}")
                return
                
            # 尝试导入transformers
            try:
                import transformers
                from transformers import AutoTokenizer, AutoModel
                has_transformers = True
                print("使用 transformers 库加载模型")
            except ImportError:
                has_transformers = False
                print("transformers 库不可用，尝试使用 SentenceTransformer")
            
            
            if has_transformers and "/" in self.model_name:
                try:
                    download_kwargs = {}
                    
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name, **download_kwargs)
                    model = AutoModel.from_pretrained(self.model_name, **download_kwargs)
                    
                    # 将transformers模型封装到类似SentenceTransformer的接口
                    class TransformersWrapper:
                        def __init__(self, model, tokenizer, device):
                            self.model = model.to(device)
                            self.tokenizer = tokenizer
                            self.device = device
                            
                        def encode(self, texts, convert_to_numpy=True):
                            if isinstance(texts, str):
                                texts = [texts]
                                
                            # 使用tokenizer处理文本
                            encoded_input = self.tokenizer(texts, padding=True, truncation=True, 
                                                       return_tensors='pt').to(self.device)
                            
                            # 获取模型输出
                            with torch.no_grad():
                                outputs = self.model(**encoded_input)
                            
                            # 获取[CLS]标记的embeddings作为句子表示
                            sentence_embeddings = outputs.last_hidden_state[:, 0]
                            
                            # 标准化embeddings
                            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                            
                            if convert_to_numpy:
                                return sentence_embeddings.cpu().numpy()
                            else:
                                return sentence_embeddings
                    
                    self.model = TransformersWrapper(model, tokenizer, self.device)
                    print(f"向量编码模型加载成功，使用设备: {self.device}")
                    return
                except Exception as e:
                    print(f"使用transformers加载模型失败，尝试使用SentenceTransformer: {str(e)}")
            
            # 尝试使用SentenceTransformer加载，设置禁用SSL验证的选项
            sentence_transformer_kwargs = {}
            
            self.model = SentenceTransformer(self.model_name, device=self.device, **sentence_transformer_kwargs)
            print(f"向量编码模型加载成功，使用设备: {self.device}")
        except Exception as e:
            logging.error(f"加载编码模型时出错: {str(e)}")
            print(f"无法加载编码模型，将使用随机向量: {str(e)}")
            self.model = None
    
    def get_entity_embedding(self, entity_text, entity_type=None):
        """获取实体的向量编码
        
        Args:
            entity_text: 实体文本
            entity_type: 实体类型 (可选)
            
        Returns:
            numpy.ndarray: 实体的向量表示
        """
        # 检查缓存
        cache_key = f"{entity_text}_{entity_type}" if entity_type else entity_text
        if cache_key in self.entity_embeddings:
            return self.entity_embeddings[cache_key]
        
        # 生成新的编码
        if self.model:
            try:
                # 如果有类型信息，将其添加到文本中增强语义
                text_to_encode = entity_text
                if entity_type:
                    text_to_encode = f"{entity_type}: {entity_text}"
                
                # 获取编码
                embedding = self.model.encode(text_to_encode, convert_to_numpy=True)
                
                # 缓存结果
                self.entity_embeddings[cache_key] = embedding
                return embedding
            except Exception as e:
                logging.warning(f"获取实体 '{entity_text}' 的向量编码时出错: {str(e)}")
        
        # 如果模型加载失败或编码出错，返回随机向量
        random_vector = np.random.normal(0, 0.1, self.embedding_dim)
        random_vector = random_vector / np.linalg.norm(random_vector)  # 归一化
        self.entity_embeddings[cache_key] = random_vector
        return random_vector
    
    def get_relation_embedding(self, source, target, relation_type=None):
        """获取关系的向量编码
        
        Args:
            source: 源实体
            target: 目标实体
            relation_type: 关系类型 (可选)
            
        Returns:
            numpy.ndarray: 关系的向量表示
        """
        # 生成缓存键
        cache_key = f"{source}__{target}"
        if relation_type:
            cache_key = f"{cache_key}__{relation_type}"
            
        # 检查缓存
        if cache_key in self.relation_embeddings:
            return self.relation_embeddings[cache_key]
        
        if self.model:
            try:
                # 构建关系描述
                relation_text = f"{source} {relation_type if relation_type else 'relates to'} {target}"
                
                # 获取编码
                embedding = self.model.encode(relation_text, convert_to_numpy=True)
                
                # 缓存结果
                self.relation_embeddings[cache_key] = embedding
                return embedding
            except Exception as e:
                logging.warning(f"获取关系的向量编码时出错: {str(e)}")
    
        random_vector = np.random.normal(0, 0.1, self.embedding_dim)
        random_vector = random_vector / np.linalg.norm(random_vector)  
        self.relation_embeddings[cache_key] = random_vector
        return random_vector
    
    def compute_similarity(self, embedding1, embedding2):
        """计算两个向量之间的余弦相似度
        
        Args:
            embedding1: 第一个向量
            embedding2: 第二个向量
            
        Returns:
            float: 余弦相似度 (-1到1之间)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        # 确保向量已归一化
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        embedding1_norm = embedding1 / norm1
        embedding2_norm = embedding2 / norm2
        
        # 计算余弦相似度
        return np.dot(embedding1_norm, embedding2_norm)
    
    def find_similar_entities(self, query_text, entity_list, top_k=5):
        """查找与查询文本最相似的实体
        
        Args:
            query_text: 查询文本
            entity_list: 实体列表，每个元素是(实体文本, 实体类型)元组
            top_k: 返回前k个最相似的实体
            
        Returns:
            list: 前k个最相似实体的(实体文本, 相似度)元组列表
        """
        if not self.model:
            return []
            
        # 获取查询文本的编码
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        
        # 计算与每个实体的相似度
        similarities = []
        for entity_text, entity_type in entity_list:
            entity_embedding = self.get_entity_embedding(entity_text, entity_type)
            similarity = self.compute_similarity(query_embedding, entity_embedding)
            similarities.append((entity_text, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个
        return similarities[:top_k]
    
    def reduce_dimensions(self, embeddings, n_components=2):
        """降维，将高维向量映射到2D或3D空间用于可视化
        
        Args:
            embeddings: 需要降维的向量列表
            n_components: 目标维度 (通常是2或3)
            
        Returns:
            numpy.ndarray: 降维后的向量数组
        """
        if len(embeddings) <= 1:
            # 不足以做PCA
            if n_components == 2:
                return np.array([[0, 0]] * len(embeddings))
            else:
                return np.array([[0, 0, 0]] * len(embeddings))
                
        pca = PCA(n_components=n_components)
        return pca.fit_transform(embeddings)
    
    def visualize_embeddings(self, entity_texts, entity_types=None, output_file=None):
        """可视化实体编码的2D分布
        
        Args:
            entity_texts: 实体文本列表
            entity_types: 实体类型列表 (可选)
            output_file: 输出文件路径 (可选)
        """
        if not entity_texts:
            return
            
        
        if entity_types is None:
            entity_types = [None] * len(entity_texts)
        elif len(entity_types) != len(entity_texts):
            entity_types = entity_types + [None] * (len(entity_texts) - len(entity_types))
        
        # 获取所有实体的编码
        embeddings = []
        for text, type_ in zip(entity_texts, entity_types):
            embeddings.append(self.get_entity_embedding(text, type_))
        
        # 降维到2D
        embeddings_2d = self.reduce_dimensions(embeddings, n_components=2)
        
        # 根据实体类型设置颜色
        unique_types = list(set([t for t in entity_types if t is not None]))
        color_map = {}
        for i, t in enumerate(unique_types):
            color_map[t] = plt.cm.tab10(i % 10)
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        for i, (text, type_) in enumerate(zip(entity_texts, entity_types)):
            x, y = embeddings_2d[i]
            color = color_map.get(type_, 'gray')
            plt.scatter(x, y, color=color)
            plt.annotate(text, (x, y), fontsize=8)
        
        # 添加图例
        for t, c in color_map.items():
            plt.scatter([], [], color=c, label=t)
        plt.legend()
        
        # 添加标题和标签
        plt.title('Entity Embeddings Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()
        
        # 保存或显示图表
        if output_file:
            directory = os.path.dirname(output_file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(output_file, dpi=300)
            print(f"Embedding visualization saved to: {output_file}")
        else:
            plt.show()
    
    def save_embeddings(self, output_file=None):
        """保存所有编码向量到文件
        
        Args:
            output_file: 输出文件路径 (可选)
        """
        if not output_file and self.path_manager:
            kg_dir = self.path_manager.get_path('knowledge_graph')
            output_file = os.path.join(kg_dir, "entity_embeddings.npz")
        
        if not output_file:
            output_file = "entity_embeddings.npz"
            
        # 创建目录（如果不存在）
        directory = os.path.dirname(output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # 将编码保存为NumPy压缩文件
        np.savez_compressed(
            output_file,
            entity_embeddings=self.entity_embeddings,
            relation_embeddings=self.relation_embeddings
        )
        print(f"Embeddings saved to: {output_file}")
        
        # 保存元数据
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "entity_count": len(self.entity_embeddings),
            "relation_count": len(self.relation_embeddings)
        }
        metadata_file = os.path.splitext(output_file)[0] + "_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def load_embeddings(self, input_file):
        """从文件加载编码向量
        
        Args:
            input_file: 输入文件路径
        """
        try:
            data = np.load(input_file, allow_pickle=True)
            self.entity_embeddings = data['entity_embeddings'].item()
            self.relation_embeddings = data['relation_embeddings'].item()
            print(f"Loaded {len(self.entity_embeddings)} entity embeddings and "
                  f"{len(self.relation_embeddings)} relation embeddings")
            return True
        except Exception as e:
            logging.error(f"加载编码向量时出错: {str(e)}")
            print(f"无法加载编码向量: {str(e)}")
            return False


# 如果直接运行此文件，执行简单测试
if __name__ == "__main__":
    print("Testing Embedding Manager...")
    
    # 初始化编码管理器
    manager = EmbeddingManager()
    
    # 测试实体编码
    test_entities = [
        ("autophagy", "PROCESS"),
        ("Treg", "CELL_TYPE"),
        ("FOXP3", "GENE"),
        ("inflammation", "DISEASE")
    ]
    
    embeddings = []
    for entity, type_ in test_entities:
        embedding = manager.get_entity_embedding(entity, type_)
        print(f"Entity: {entity}, Type: {type_}, Embedding shape: {embedding.shape}")
        embeddings.append(embedding)
    
    # 测试相似度计算
    sim = manager.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between 'autophagy' and 'Treg': {sim:.4f}")
    
    # 可视化测试
    manager.visualize_embeddings(
        [e[0] for e in test_entities],
        [e[1] for e in test_entities],
        "test_embeddings.png"
    )
    
    print("Test completed!") 