# 文献分析与知识图谱构建系统

## 项目概述

该项目是一个专门用于分析自噬(Autophagy)和调节性T细胞(Treg)相关文献的工具，通过文本挖掘和机器学习技术构建知识图谱。系统采用模块化设计，集成了实体识别、关系提取、向量表示和知识图谱可视化等功能，并引入了基于TinyBERT模型的关系分类功能，可以预测实体间的关系类型。

## 系统架构

整个系统由以下几个核心模块组成：

1. **数据处理与分析**：加载和处理文献数据，进行基本统计分析
2. **实体识别**：基于词典和规则的方法识别文本中的生物医学实体
3. **向量表示**：使用预训练语言模型为实体生成语义向量
4. **关系提取与分类**：从文本中提取实体关系，并使用训练好的模型预测关系类型
5. **知识图谱构建与可视化**：构建生物医学知识图谱并提供多种可视化方式

## 文件结构与功能

### 核心执行文件

- `simple_analysis.py`: 系统的主入口，执行文献分析和知识图谱构建的完整流程
- `train_main.py`: 关系分类模型的训练脚本, 输入的文件需要包含source_entity,target_entity,relation_type这些列名
- `test_relation_trainer.py`: 测试TinyBERT模型初始化和配置的辅助脚本

### 核心功能模块

- `knowledge_graph.py`: 知识图谱构建器，负责实体识别、关系提取和图谱生成
- `embedding_manager.py`: 向量编码管理器，为实体和关系生成向量表示
- `relation_trainer.py`: 关系分类器，用于训练和预测实体间关系类型

### 数据与配置

- `config/`: 包含系统配置文件，如实体词典、关系类型和可视化设置
- `data/`: 存放输入数据（如自定义训练文献CSV文件/搜索文献后的CSV文件）和中间处理结果
- `models/`: 包含预训练模型和训练好的关系分类模型
- `output/`: 存储知识图谱和可视化结果

### 辅助组件

- `utils/`: 工具函数和辅助类，如路径管理器
- `nltk/`: 自然语言处理资源和数据
- `logs/`: 运行日志

## 运行流程

### 基本分析流程

1. **初始化配置**：加载配置文件和设置运行环境
2. **数据加载**：读取文献CSV文件
3. **基本分析**：计算文献统计信息，如年份分布、期刊分布
4. **实体与关系提取**：识别自噬和Treg相关实体，提取它们之间的关系
5. **构建知识图谱**：创建网络图结构，添加节点和边
6. **可视化输出**：生成交互式HTML可视化

### 关系分类模型训练流程

1. **准备训练数据**：从CSV加载或使用预定义的关系示例
2. **初始化模型**：基于TinyBERT创建序列分类模型
3. **模型训练**：通过梯度下降优化模型参数
4. **模型评估**：使用验证集评估模型性能
5. **模型保存**：将训练好的模型保存到指定目录

## 核心功能详解

### 1. 实体识别

系统使用词典和规则方法从文本中识别以下类型的生物医学实体：

- 基因/蛋白质（如FOXP3、LC3、mTOR）
- 化学物质/细胞因子（如IL-10、TGF-β）
- 疾病/症状（如炎症、自身免疫）
- 细胞类型（如Treg、T细胞）

### 2. 向量表示

系统使用TinyBERT模型为识别出的实体生成语义向量，实现以下功能：

- 计算实体间的语义相似度
- 基于相似度推断潜在关系
- 通过降维可视化实体语义空间

### 3. 关系分类

系统训练了一个关系分类模型，能够预测实体间的六种关系类型：

- `positive_regulation`: 正向调控（如激活、促进）
- `negative_regulation`: 负向调控（如抑制、减少）
- `regulation`: 一般调控（无明确方向）
- `association`: 相关性（无因果关系）
- `conversion`: 转化/转变关系
- `DEFAULT`: 默认关系类型

### 4. 知识图谱构建与可视化

系统提供多种知识图谱格式和可视化方式：

- 交互式HTML图谱（使用PyVis）
- ECharts交互式可视化
- GEXF格式导出（兼容Gephi）

## 使用示例

### 基本使用

在已训练好模型的基础上使用

```bash
# 运行基本分析和知识图谱构建
python simple_analysis.py -i data/Autophagy_Treg_literature.csv
```

### 关系模型训练

```bash
# 使用自定义关系数据训练模型
python train_main.py -d data/relations_data.csv -e 10 -t
```

## 系统特点与优势

1. **模块化设计**：各功能模块独立，便于扩展和维护
2. **多种可视化**：支持多种图谱可视化格式，满足不同需求
3. **向量语义**：利用预训练语言模型捕捉实体语义关系
4. **关系预测**：通过微调TinyBERT实现准确的关系分类
5. **可配置性**：大部分功能可通过配置文件调整

## 技术栈

- **Python**：核心编程语言
- **NetworkX**：网络图数据结构
- **PyVis/ECharts**：交互式可视化
- **NLTK**：自然语言处理
- **Transformers/PyTorch**：深度学习与向量表示
- **Pandas**：数据处理
## 感谢华为小组蒸馏的bert模型，膜拜
原模型位于huggingface,模型名： huawei-noah/TinyBERT_General_4L_312D model
论文参考：https://doi.org/10.48550/arXiv.1909.10351
