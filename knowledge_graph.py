import pandas as pd
import networkx as nx
from pyvis.network import Network
import json
import logging
from datetime import datetime
import os
import re
from utils.path_manager import PathManager
try:
    from embedding_manager import EmbeddingManager
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("Warning: EmbeddingManager not available. Vector encoding features will be disabled.")

class KnowledgeGraphBuilder:
    """知识图谱构建类"""
    
    def __init__(self, config_file="config/autophagy_treg_config.yaml"):
        """初始化知识图谱构建器"""
        self.path_manager = PathManager(config_file)
        self.setup_logging()
        
        self.G = nx.Graph()
        
        # 添加专业词典
        self.bio_dict = {
            "CHEMICAL": {
                "LPS", "IL-1α", "IL-1β", "IL-2", "IL-4", "IL-6", "IL-8", "IL-10", "IL-12",
                "IL-13", "IL-17", "IL-18", "IL-21", "IL-22", "IL-23",
                "TNF-α", "TNF-β", "IFN-α", "IFN-β", "IFN-γ",
                "TGF-β", "TGF-β1", "TGF-β2", "TGF-β3",
                "peptide", "protein", "antibody",
                "autophagy", "autophagic", "macroautophagy", "mitophagy",
                "LC3", "Beclin", "mTOR", "AMPK", "p62", "ULK1", "ATG", "SQSTM1",
                "autophagosome", "autolysosome", "lysosome", "phagosome", "phagophore",
                "rapamycin", "bafilomycin", "3-methyladenine", "chloroquine",
                "IL-10", "TGF-beta", "TGF-β", "CTLA-4", "PD-1", "GITR",
                "CD25", "CD28", "CD4", "CD8", "cytokine", "chemokine"
            },
            "GENE": {
                "FOXP3", "NF-κB", "p53", "EGFR", "ATG", "BECN1", "LC3", "MAP1LC3",
                "SQSTM1", "p62", "ULK1", "mTOR", "AMPK", "STAT1", "STAT3", "STAT6",
                "JAK1", "JAK2", "JAK3", "PIK3C3", "CTLA4", "CD25", "IL2RA",
                "IL10", "TGFB1", "TGFB2", "TGFB3", "PDCD1", "IKZF2", "IKZF4",
            },
            "DISEASE": {
                "inflammation", "cancer", "colitis", "arthritis", "diabetes",
                "autoimmune", "allergy", "asthma", "infection", "tumor",
                "malignancy", "graft-versus-host", "transplantation", "rejection",
                "lupus", "multiple sclerosis", "IBD", "inflammatory bowel disease"
            },
            "CELL_TYPE": {
                "Treg", "regulatory T cell", "regulatory T cells", "T cell", "T cells", 
                "B cell", "B cells", "macrophage", "macrophages", "dendritic cell", 
                "dendritic cells", "DC", "DCs", "neutrophil", "neutrophils", "monocyte", 
                "monocytes", "lymphocyte", "lymphocytes", "NK cell", "NK cells", 
                "natural killer", "eosinophil", "eosinophils", "basophil", "basophils", 
                "mast cell", "mast cells", "CD4+", "CD8+", "CD4+CD25+", "CD4+CD25+FOXP3+"
            }
        }
        
        # 加载更多词典
        self.load_extra_dictionaries()
        
        # 初始化向量编码管理器
        self.embedding_manager = None
        self.use_embeddings = False
        self.initialize_embedding_manager()
        
        # 初始化关系训练器（如果可用）
        self.relation_trainer = None
        try:
            from relation_trainer import RelationTrainer
            self.relation_trainer = RelationTrainer(config_file=config_file)
            print("关系训练器初始化成功")
        except ImportError:
            print("关系训练器不可用，将使用规则推断关系")
        
    def initialize_embedding_manager(self):
        """初始化向量编码管理器"""
        if not EMBEDDING_AVAILABLE:
            return
            
        try:
            # 检查配置中是否启用了向量编码
            if (hasattr(self.path_manager, 'config') and 
                'embedding' in self.path_manager.config and 
                self.path_manager.config['embedding'].get('enabled', False)):
                
                print("\n初始化向量编码功能...")
                
                self.embedding_manager = EmbeddingManager(
                    config_file=self.path_manager.config_file
                )
                self.use_embeddings = True
                
                # 检查是否有缓存文件
                if (self.path_manager.config['embedding'].get('cache', {}).get('enabled', False) and
                    self.path_manager.config['embedding']['cache'].get('file')):
                    
                    cache_file = self.path_manager.config['embedding']['cache']['file']
                    kg_dir = self.path_manager.get_path('knowledge_graph')
                    cache_path = os.path.join(kg_dir, cache_file)
                    
                    if os.path.exists(cache_path):
                        print(f"Loading embeddings from cache: {cache_path}")
                        self.embedding_manager.load_embeddings(cache_path)
        except Exception as e:
            logging.error(f"初始化向量编码管理器时出错: {str(e)}")
            print(f"无法初始化向量编码功能: {str(e)}")
            self.embedding_manager = None
            self.use_embeddings = False
        
    def load_extra_dictionaries(self):
        """加载额外的词典数据"""
        try:
            # 尝试从配置文件加载实体
            if hasattr(self.path_manager, 'config') and 'entities' in self.path_manager.config:
                config = self.path_manager.config
                
                # 加载自噬相关术语
                if 'autophagy_terms' in config['entities']:
                    self.bio_dict["CHEMICAL"].update(
                        [term.lower() for term in config['entities']['autophagy_terms']]
                    )
                    
                # 加载Treg相关术语
                if 'treg_terms' in config['entities']:
                    self.bio_dict["CELL_TYPE"].update(
                        [term.lower() for term in config['entities']['treg_terms']]
                    )
                    
                # 加载疾病相关术语
                if 'disease_terms' in config['entities']:
                    self.bio_dict["DISEASE"].update(
                        [term.lower() for term in config['entities']['disease_terms']]
                    )
        except Exception as e:
            print(f"加载额外词典时出错: {str(e)}")
        
    def setup_logging(self):
        """配置日志"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"knowledge_graph_{timestamp}.log")
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def extract_entities(self, text):
        """使用规则和词典提取实体"""
        # 将文本转为小写处理
        text_lower = text.lower()
        
        # 使用正则表达式查找句子
        sentences = re.split(r'[.!?]', text_lower)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        found_entities = []
        
        for sentence in sentences:
            # 为每种实体类型检查词典匹配
            for entity_type, terms in self.bio_dict.items():
                for term in terms:
                    term_lower = term.lower()
                    if term_lower in sentence:
                        # 找到实体的位置
                        start = sentence.find(term_lower)
                        end = start + len(term_lower)
                        
                        # 确保这是一个完整的词（不是更大词的一部分）
                        if ((start == 0 or not sentence[start-1].isalnum()) and 
                            (end == len(sentence) or not sentence[end].isalnum())):
                            
                            found_entities.append({
                                'text': term,
                                'label_': entity_type,
                                'start': start,
                                'end': end,
                                'sentence': sentence
                            })
        
        # 返回找到的实体列表
        return found_entities
    
    def build_graph(self, data_file=None):
        """构建知识图谱"""
        try:
            print("开始构建知识图谱...")
            
            # 使用path_manager获取分析结果文件路径
            if data_file is None:
                analysis_dir = os.path.dirname(self.path_manager.get_path('analysis', 'wordcloud'))
                data_file = os.path.join(analysis_dir, "analyzed_literature.csv")
            
            print(f"\n1. 正在加载分析数据: {data_file}")
            df = pd.read_csv(data_file)
            print(f"已加载 {len(df)} 篇文献的数据")
            
            # 添加节点和边
            print("\n2. 正在提取实体关系...")
            total = len(df)
            relations_count = 0
            
            for i, row in enumerate(df.iterrows(), 1):
                print(f"处理进度: {i}/{total} 篇文献", end='\r')
                if pd.isna(row[1]["Abstract"]):
                    continue
                    
                relations = self.extract_relations(row[1]["Abstract"])
                relations_count += len(relations)
                
                for rel in relations:
                    # 添加节点
                    self.G.add_node(
                        rel["source"],
                        node_type=rel["source_type"]
                    )
                    
                    self.G.add_node(
                        rel["target"],
                        node_type=rel["target_type"]
                    )
                    
                    # 添加边
                    if self.G.has_edge(rel["source"], rel["target"]):
                        self.G[rel["source"]][rel["target"]]["weight"] += 1
                    else:
                        # 确定关系类型
                        # 基于实体类型猜测可能的关系类型
                        relation_type = self._determine_relation_type(rel["source_type"], rel["target_type"], rel["source"], rel["target"])
                        
                        self.G.add_edge(
                            rel["source"],
                            rel["target"],
                            weight=1,
                            relation_type=relation_type
                        )
            
            # 如果使用向量编码，为节点添加向量表示
            if self.use_embeddings and self.embedding_manager:
                self.add_node_embeddings()
                
                # 可选：基于向量相似度添加额外的边
                if (hasattr(self.path_manager, 'config') and 
                    'embedding' in self.path_manager.config and 
                    'similarity' in self.path_manager.config['embedding'] and
                    self.path_manager.config['embedding']['similarity'].get('infer_new_relations', False)):
                    
                    self.add_similarity_based_edges()
                
                # 保存编码到缓存
                if (hasattr(self.path_manager, 'config') and 
                    'embedding' in self.path_manager.config and 
                    'cache' in self.path_manager.config['embedding'] and
                    self.path_manager.config['embedding']['cache'].get('enabled', False)):
                    
                    self.embedding_manager.save_embeddings()
                
                # 可视化向量编码
                if (hasattr(self.path_manager, 'config') and 
                    'embedding' in self.path_manager.config and 
                    'visualization' in self.path_manager.config['embedding'] and 
                    self.path_manager.config['embedding']['visualization'].get('enabled', False)):
                    
                    output_file = self.path_manager.config['embedding']['visualization'].get(
                        'output_file', 'output/knowledge_graph/embeddings_visualization.png'
                    )
                    self.visualize_embeddings(output_file)
            
            print(f"\n\n3. 知识图谱构建完成:")
            print(f"- 节点数量: {len(self.G.nodes)}")
            print(f"- 边的数量: {len(self.G.edges)}")
            print(f"- 关系总数: {relations_count}")
            
        except Exception as e:
            print(f"\n构建图谱时出错: {str(e)}")
            logging.error(f"构建图谱时出错: {str(e)}")
    
    def add_node_embeddings(self):
        """为图中所有节点添加向量表示"""
        if not self.use_embeddings or not self.embedding_manager:
            return
            
        print("\n为实体生成向量表示...")
        for node, data in self.G.nodes(data=True):
            # 获取向量编码
            node_type = data.get('node_type', 'DEFAULT')
            embedding = self.embedding_manager.get_entity_embedding(node, node_type)
            
            # 将向量存储在节点属性中
            self.G.nodes[node]['embedding'] = embedding.tolist()  # 转换为列表以便JSON序列化
    
    def add_similarity_based_edges(self):
        """基于向量相似度添加额外的边"""
        if not self.use_embeddings or not self.embedding_manager:
            return
            
        # 获取相似度阈值
        threshold = 0.7  # 默认阈值
        if (hasattr(self.path_manager, 'config') and 
            'embedding' in self.path_manager.config and 
            'similarity' in self.path_manager.config['embedding']):
            threshold = self.path_manager.config['embedding']['similarity'].get('threshold', 0.7)
        
        print(f"\n基于向量相似度(阈值={threshold})推断新的关系...")
        new_edges = 0
        
        # 获取所有节点
        nodes = list(self.G.nodes(data=True))
        
        # 计算节点对之间的相似度
        for i in range(len(nodes)):
            node1, data1 = nodes[i]
            embedding1 = data1.get('embedding')
            if embedding1 is None:
                continue
                
            for j in range(i+1, len(nodes)):
                node2, data2 = nodes[j]
                embedding2 = data2.get('embedding')
                if embedding2 is None:
                    continue
                
                # 跳过已有的边
                if self.G.has_edge(node1, node2):
                    continue
                
                # 计算相似度
                similarity = self.embedding_manager.compute_similarity(
                    embedding1, embedding2
                )
                
                # 如果相似度超过阈值且节点类型不同，添加新边
                if similarity > threshold and data1.get('node_type') != data2.get('node_type'):
                    self.G.add_edge(
                        node1, node2, 
                        weight=1, 
                        similarity=similarity,
                        relation_type='similarity_inferred'
                    )
                    new_edges += 1
        
        print(f"- 添加了 {new_edges} 个基于相似度的新关系")
    
    def visualize_embeddings(self, output_file=None):
        """可视化实体的向量表示"""
        if not self.use_embeddings or not self.embedding_manager:
            return
            
        # 确保输出目录存在
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
        print("\n可视化实体向量编码...")
        
        # 收集实体和类型
        entity_texts = []
        entity_types = []
        for node, data in self.G.nodes(data=True):
            entity_texts.append(node)
            entity_types.append(data.get('node_type'))
        
        # 生成可视化
        self.embedding_manager.visualize_embeddings(
            entity_texts, entity_types, output_file
        )
        
        print(f"- 向量可视化已保存到: {output_file}")
    
    def extract_relations(self, text):
        """基于共现提取实体关系"""
        # 提取实体
        entities = self.extract_entities(text)
        relations = []
        
        # 根据句子分组实体
        sentence_entities = {}
        for entity in entities:
            sentence = entity['sentence']
            if sentence not in sentence_entities:
                sentence_entities[sentence] = []
            sentence_entities[sentence].append(entity)
        
        # 对于每个句子中的实体，构建实体间的关系
        for sentence, entities_in_sentence in sentence_entities.items():
            for i in range(len(entities_in_sentence)):
                for j in range(i+1, len(entities_in_sentence)):
                    ent1 = entities_in_sentence[i]
                    ent2 = entities_in_sentence[j]
                    
                    # 只在不同类型的实体之间建立关系
                    if ent1['label_'] != ent2['label_']:
                        relations.append({
                            "source": ent1['text'],
                            "source_type": ent1['label_'],
                            "target": ent2['text'],
                            "target_type": ent2['label_']
                        })
        
        return relations
    
    def visualize_graph(self, output_file=None):
        """可视化知识图谱"""
        try:
            # 检查是否有节点
            if len(self.G.nodes) == 0:
                print("图谱为空，无法生成可视化")
                return
            
            # 使用path_manager获取knowledge_graph目录路径
            kg_dir = self.path_manager.get_path('knowledge_graph')
            if output_file is None:
                output_file = os.path.join(kg_dir, "knowledge_graph.html")
            
            # 获取绝对路径
            abs_path = os.path.abspath(output_file)
            print(f"\n文件将保存到: {abs_path}")
            
            print("\n4. 正在生成可视化图谱...")
            
            # 获取节点样式配置
            node_border_width = self.get_node_style("border_width")
            node_border_width_selected = self.get_node_style("border_width_selected")
            node_shape = self.get_node_style("shape")
            
            # 获取边样式配置
            edge_smooth = self.get_edge_style("smooth")
            
            # 创建PyVis网络对象，添加更多配置
            net = Network(
                height="800px",
                width="100%",
                bgcolor="#ffffff",
                font_color="black",
                directed=False,
                select_menu=True,
                filter_menu=True,
                notebook=False
            )
            
            # 添加加载提示和进度条的HTML
            loading_html = """
            <div id="loading" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                 background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.5);">
                <h3>Loading Knowledge Graph...</h3>
                <p>Number of nodes: {nodes_count}</p>
                <p>Number of edges: {edges_count}</p>
                <p>Please wait...</p>
            </div>
            <script>
                window.addEventListener('load', function() {{
                    document.getElementById('loading').style.display = 'none';
                }});
            </script>
            """.format(nodes_count=len(self.G.nodes), edges_count=len(self.G.edges))
            
            # 优化性能的配置
            net.set_options("""
            {
                "physics": {
                    "enabled": true,
                    "stabilization": {
                        "iterations": 50,
                        "updateInterval": 100,
                        "fit": true
                    },
                    "barnesHut": {
                        "gravitationalConstant": -1000,
                        "springLength": 150,
                        "springConstant": 0.02,
                        "damping": 0.09,
                        "avoidOverlap": 0.5
                    }
                },
                "nodes": {
                    "scaling": {
                        "min": 5,
                        "max": 30
                    },
                    "shape": "%s"
                },
                "edges": {
                    "smooth": %s,
                    "scaling": {
                        "min": 1,
                        "max": 3
                    }
                },
                "interaction": {
                    "hideEdgesOnDrag": true,
                    "hideNodesOnDrag": false,
                    "navigationButtons": true,
                    "keyboard": true,
                    "hover": true
                }
            }
            """ % (node_shape, str(edge_smooth).lower()))
            
            # 添加节点，设置更多属性
            print("- Adding nodes...")
            for node in self.G.nodes(data=True):
                size = self.G.degree(node[0]) * 2  # 减小节点大小系数
                net.add_node(
                    node[0],
                    label=node[0],
                    title=f"Type: {node[1]['node_type']}\nDegree: {self.G.degree(node[0])}",
                    color=self.get_node_color(node[1]['node_type']),
                    size=size,
                    borderWidth=node_border_width,
                    borderWidthSelected=node_border_width_selected
                )
            
            # 添加边，设置更多属性
            print("- Adding edges...")
            for edge in self.G.edges(data=True):
                width = edge[2]['weight']  # 简化边的宽度
                net.add_edge(
                    edge[0],
                    edge[1],
                    value=width,
                    title=f"Weight: {edge[2]['weight']}"
                )
            
            # 保存文件时插入加载提示
            try:
                html_content = net.generate_html()
                # 在body标签后插入加载提示
                html_content = html_content.replace('<body>', f'<body>\n{loading_html}')
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"Interactive graph saved to: {abs_path}")
                print("Note: First loading may take some time, please be patient")
            except Exception as e:
                print(f"Error saving HTML file: {str(e)}")
            
        except Exception as e:
            print(f"\nError visualizing graph: {str(e)}")
            logging.error(f"Error visualizing graph: {str(e)}")
    
    def get_node_color(self, node_type):
        """获取节点颜色"""
        # 默认颜色方案
        default_colors = {
            "GENE": "#ff7f0e",     # 橙色    
            "CHEMICAL": "#2ca02c", # 绿色 化学物质
            "DISEASE": "#d62728",  # 红色
            "CELL_TYPE": "#9467bd", # 紫色
            "DEFAULT": "#1f77b4"   # 蓝色  #未知未识别
        }
        
        # 尝试从配置文件中读取颜色
        try:
            if (hasattr(self.path_manager, 'config') and 
                'visualization' in self.path_manager.config and 
                'color_scheme' in self.path_manager.config['visualization'] and
                'entity_types' in self.path_manager.config['visualization']['color_scheme']):
                
                config_colors = self.path_manager.config['visualization']['color_scheme']['entity_types']
                return config_colors.get(node_type, config_colors.get("DEFAULT", default_colors["DEFAULT"]))
        except Exception as e:
            logging.warning(f"读取节点颜色配置时出错: {str(e)}，使用默认颜色")
            
        # 如果配置读取失败或没有该类型的颜色，返回默认颜色
        return default_colors.get(node_type, default_colors["DEFAULT"])
    
    def get_edge_color(self, edge_type=None):
        """获取边的颜色"""
        # 默认边颜色方案
        default_colors = {
            "positive_regulation": "#00a300",  # 深绿色 - 激活/诱导关系
            "negative_regulation": "#cc0000",  # 深红色 - 抑制关系
            "regulation": "#0066cc",           # 蓝色 - 一般调控关系
            "DEFAULT": "#999999"               # 灰色 - 默认/未分类关系
        }
        
        # 如果没有指定类型，返回默认颜色
        if edge_type is None:
            return default_colors["DEFAULT"]
            
        # 尝试从配置文件中读取颜色
        try:
            if (hasattr(self.path_manager, 'config') and 
                'visualization' in self.path_manager.config and 
                'color_scheme' in self.path_manager.config['visualization'] and
                'relation_types' in self.path_manager.config['visualization']['color_scheme']):
                
                config_colors = self.path_manager.config['visualization']['color_scheme']['relation_types']
                return config_colors.get(edge_type, config_colors.get("DEFAULT", default_colors["DEFAULT"]))
        except Exception as e:
            logging.warning(f"读取边颜色配置时出错: {str(e)}，使用默认颜色")
            
        # 如果配置读取失败或没有该类型的颜色，返回默认颜色
        return default_colors.get(edge_type, default_colors["DEFAULT"])
    
    def get_node_style(self, style_key=None):
        """获取节点样式设置"""
        # 默认节点样式
        default_styles = {
            "border_width": 2,
            "border_width_selected": 3,
            "shape": "dot"
        }
        
        # 如果没有指定样式键，返回所有默认样式
        if style_key is None:
            return default_styles
            
        # 尝试从配置文件中读取样式
        try:
            if (hasattr(self.path_manager, 'config') and 
                'visualization' in self.path_manager.config and 
                'styles' in self.path_manager.config['visualization'] and
                'node' in self.path_manager.config['visualization']['styles']):
                
                config_styles = self.path_manager.config['visualization']['styles']['node']
                if style_key in config_styles:
                    return config_styles[style_key]
        except Exception as e:
            logging.warning(f"读取节点样式配置时出错: {str(e)}，使用默认样式")
            
        # 如果配置读取失败或没有该样式，返回默认样式
        return default_styles.get(style_key, None)
    
    def get_edge_style(self, style_key=None):
        """获取边样式设置"""
        # 默认边样式
        default_styles = {
            "smooth": False,
            "arrows": False,
            "dashes": False
        }
        
        # 如果没有指定样式键，返回所有默认样式
        if style_key is None:
            return default_styles
            
        # 尝试从配置文件中读取样式
        try:
            if (hasattr(self.path_manager, 'config') and 
                'visualization' in self.path_manager.config and 
                'styles' in self.path_manager.config['visualization'] and
                'edge' in self.path_manager.config['visualization']['styles']):
                
                config_styles = self.path_manager.config['visualization']['styles']['edge']
                if style_key in config_styles:
                    return config_styles[style_key]
        except Exception as e:
            logging.warning(f"读取边样式配置时出错: {str(e)}，使用默认样式")
            
        # 如果配置读取失败或没有该样式，返回默认样式
        return default_styles.get(style_key, None)
    
    def export_graph(self, output_file=None):
        """导出图谱为Gephi格式"""
        try:
            # 使用path_manager获取knowledge_graph目录路径
            kg_dir = self.path_manager.get_path('knowledge_graph')
            if output_file is None:
                output_file = os.path.join(kg_dir, "knowledge_graph.gexf")
            
            # 创建一个全新的图，避免使用现有的图中可能存在的复杂数据结构
            export_graph = nx.Graph()
            
            # 添加节点和基本属性，但不包括会导致问题的viz属性
            for node, data in self.G.nodes(data=True):
                export_graph.add_node(node, label=str(node), node_type=data.get('node_type', 'DEFAULT'))
                
                # 设置颜色属性 - 使用与测试脚本相同的结构
                node_type = data.get('node_type', 'DEFAULT')
                color = self.get_node_color(node_type)
                
                if isinstance(color, str) and color.startswith('#') and len(color) == 7:
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                else:
                    r, g, b = 128, 128, 128  # 默认灰色
                
                # 简单化viz结构，与测试案例保持一致
                export_graph.nodes[node]['viz'] = {
                    'color': {'r': r, 'g': g, 'b': b, 'a': 1.0},  # 修复了颜色通道设置错误
                    'size': 10.0,
                    'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                }
            
            # 添加边和权重，并设置边的颜色
            for u, v, data in self.G.edges(data=True):
                weight = data.get('weight', 1)
                
                # 获取边的关系类型并设置颜色
                relation_type = data.get('relation_type', 'DEFAULT')
                color = self.get_edge_color(relation_type)
                
                # 解析十六进制颜色代码
                if isinstance(color, str) and color.startswith('#') and len(color) == 7:
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                else:
                    r, g, b = 153, 153, 153  # 默认灰色
                
                # 添加边及其属性
                export_graph.add_edge(
                    u, v, 
                    weight=float(weight),
                    relation_type=relation_type
                )
                
                # 为边添加viz属性
                export_graph[u][v]['viz'] = {
                    'color': {'r': r, 'g': g, 'b': b, 'a': 0.8},
                    'thickness': min(weight, 5),  # 边的粗细与权重相关
                    'shape': 'solid'  # 可选: solid, dash, dotted
                }
            
            # 导出为GEXF格式，使用最简单的方式
            print(f"\n尝试导出到: {output_file}")
            nx.write_gexf(export_graph, output_file)
            print(f"Gephi格式文件已保存到: {output_file}")
            
        except Exception as e:
            print(f"导出图谱时出错: {str(e)}")
            logging.error(f"导出图谱时出错: {str(e)}")
            
            # 尝试导出更简单的版本
            try:
                print("尝试导出简化版本，不包含可视化属性...")
                simple_graph = nx.Graph()
                
                # 仅添加节点和基本属性，不包含任何viz
                for node, data in self.G.nodes(data=True):
                    attrs = {}
                    for key, value in data.items():
                        if key != 'viz':
                            attrs[key] = value
                    simple_graph.add_node(node, **attrs)
                
                # 添加边
                for u, v, data in self.G.edges(data=True):
                    simple_graph.add_edge(u, v, weight=1.0)
                
                # 导出简化版本
                simple_output = os.path.join(kg_dir, "knowledge_graph_simple.gexf")
                nx.write_gexf(simple_graph, simple_output)
                print(f"简化的Gephi格式文件已保存到: {simple_output}")
            except Exception as e2:
                print(f"导出简化图谱时出错: {str(e2)}")
                logging.error(f"导出简化图谱时出错: {str(e2)}")
    
    def export_for_echarts(self, output_file=None):
        """导出为ECharts格式"""
        try:
            if len(self.G.nodes) == 0:
                print("Graph is empty, cannot generate visualization")
                return
            
            print("\nGenerating ECharts visualization...")
            
            # 只保留权重大于1的边和相关节点
            important_edges = [(u, v, d) for u, v, d in self.G.edges(data=True) if d['weight'] > 1]
            important_nodes = set()
            for u, v, _ in important_edges:
                important_nodes.add(u)
                important_nodes.add(v)
            
            # 准备节点数据
            nodes = []
            for node, attrs in self.G.nodes(data=True):
                if node in important_nodes:
                    # 如果节点没有类型，设置默认类型
                    node_type = attrs.get('node_type', 'UNKNOWN')
                    
                    # 获取节点颜色并进行检查
                    color = self.get_node_color(node_type)
                    if not (isinstance(color, str) and color.startswith('#') and len(color) == 7):
                        color = "#999999"  # 默认灰色
                        
                    nodes.append({
                        'name': node,
                        'category': node_type,
                        'symbolSize': min(max(self.G.degree(node) * 3, 10), 50),
                        'value': self.G.degree(node),
                        'itemStyle': {
                            'color': color
                        }
                    })
            
            # 准备边数据（只包含重要边）
            links = [
                {
                    'source': u,
                    'target': v,
                    'value': d['weight'],
                    'lineStyle': {
                        'width': min(d['weight'], 5),
                        'opacity': 0.5
                    }
                }
                for u, v, d in important_edges  # 只包含重要边
            ]
            
            # 定义节点类别
            categories = []
            for category_name in ['GENE', 'CHEMICAL', 'DISEASE', 'CELL_TYPE']:
                color = self.get_node_color(category_name)
                # 检查颜色是否有效
                if not (isinstance(color, str) and color.startswith('#') and len(color) == 7):
                    color = "#999999"  # 默认灰色
                
                categories.append({
                    'name': category_name, 
                    'itemStyle': {'color': color}
                })
            
            # 获取图表标题
            title_text = "Autophagy-Treg Knowledge Graph (Important Relations)"
            if hasattr(self.path_manager, 'config') and 'visualization' in self.path_manager.config:
                if 'title' in self.path_manager.config['visualization']:
                    title_text = self.path_manager.config['visualization']['title']
            
            # 优化 ECharts 配置
            option = {
                'title': {'text': title_text},
                'tooltip': {
                    'formatter': '{b}: {c}'
                },
                'legend': [{
                    'data': [c['name'] for c in categories],
                    'selected': {c['name']: True for c in categories}
                }],
                'animationDurationUpdate': 1000,  # 减少动画时间
                'animationEasingUpdate': 'linear',  # 使用线性动画
                'series': [{
                    'type': 'graph',
                    'layout': 'force',
                    'data': nodes,
                    'links': links,
                    'categories': categories,
                    'roam': True,
                    'label': {
                        'show': True,
                        'position': 'right',
                        'fontSize': 12,
                        'formatter': '{b}'  # 只显示名称
                    },
                    'force': {
                        'repulsion': 200,  # 增加斥力
                        'gravity': 0.2,    # 增加引力
                        'edgeLength': 150,  # 增加边长
                        'layoutAnimation': False  # 关闭布局动画
                    },
                    'emphasis': {
                        'focus': 'adjacency',
                        'lineStyle': {
                            'width': 10
                        }
                    }
                }]
            }
            
            # 生成HTML文件
            if output_file is None:
                output_file = os.path.join(
                    self.path_manager.get_path('knowledge_graph'),
                    'knowledge_graph_echarts.html'
                )
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Autophagy-Treg Knowledge Graph</title>
                <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
                <style>
                    #main {{ height: 100vh; margin: 0; }}
                    body {{ margin: 0; }}
                </style>
            </head>
            <body>
                <div id="main"></div>
                <script>
                    var myChart = echarts.init(document.getElementById('main'));
                    var option = {option};
                    myChart.setOption(option);
                    window.addEventListener('resize', function() {{
                        myChart.resize();
                    }});
                </script>
            </body>
            </html>
            """
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"ECharts graph saved to: {output_file}")
            
        except Exception as e:
            print(f"Error generating ECharts graph: {str(e)}")
            logging.error(f"Error generating ECharts graph: {str(e)}")

    def _determine_relation_type(self, source_type, target_type, source_text=None, target_text=None):
        """根据实体类型和文本确定关系类型
        
        Args:
            source_type: 源实体类型
            target_type: 目标实体类型
            source_text: 源实体文本
            target_text: 目标实体文本
            
        Returns:
            str: 关系类型，可能是 'positive_regulation', 'negative_regulation', 或 'regulation'
        """
        # 优先使用关系分类器（如果可用且有源实体和目标实体文本）
        if hasattr(self, 'relation_trainer') and self.relation_trainer and source_text and target_text:
            try:
                relation, confidence = self.relation_trainer.predict_relation(source_text, target_text)
                # 只有当置信度超过阈值时才使用预测结果
                if confidence > 0.2:
                    return relation
            except Exception as e:
                logging.warning(f"使用关系分类器预测关系时出错: {str(e)}")
        
        # 基于实体类型的组合确定关系类型
        # 以下是基于生物学知识的简单规则
        
        # 基因-细胞 互动通常是调控关系
        if (source_type == 'GENE' and target_type == 'CELL_TYPE') or \
           (source_type == 'CELL_TYPE' and target_type == 'GENE'):
            return 'regulation'
        
        # 化学物质-基因 互动通常可能是正向或负向调控
        if (source_type == 'CHEMICAL' and target_type == 'GENE') or \
           (source_type == 'GENE' and target_type == 'CHEMICAL'):
            # 这里简单随机分配，实际应用中应该基于文本内容分析
            import random
            return random.choice(['positive_regulation', 'negative_regulation'])
        
        # 化学物质-细胞 互动通常可能是正向或负向调控  
        if (source_type == 'CHEMICAL' and target_type == 'CELL_TYPE') or \
           (source_type == 'CELL_TYPE' and target_type == 'CHEMICAL'):
            # 简单随机分配
            import random
            return random.choice(['positive_regulation', 'negative_regulation'])
        
        # 疾病相关的关系
        if source_type == 'DISEASE' or target_type == 'DISEASE':
            return 'regulation'  # 通常是一般调控关系
        
        # 默认关系类型
        return 'DEFAULT'

def main():
    print("Starting knowledge graph building process...")
    
    # 构建知识图谱
    builder = KnowledgeGraphBuilder()
    
    # 构建图谱
    builder.build_graph()
    
    # 只有在图谱构建成功时才继续
    if len(builder.G.nodes) > 0:
        # 导出为ECharts格式（更快的可视化）
        builder.export_for_echarts()
        
        # 导出为Gephi格式（用于深入分析）
        builder.export_graph()
        
        print("\nAll processing completed!")
        print(f"- Number of nodes: {len(builder.G.nodes)}")
        print(f"- Number of edges: {len(builder.G.edges)}")
    else:
        print("\nGraph building failed, no visualization files generated")

if __name__ == "__main__":
    main() 