#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Analysis Tool for Autophagy and Regulatory T Cell Literature

This script performs basic analysis on literature data related to autophagy and Tregs,
focusing on entity extraction and knowledge graph construction without generating visualizations.

Note: This version uses a rule-based and dictionary-based approach for entity extraction
instead of the biolinkbert model, making it more lightweight and easier to run.
"""

import os
import re
import yaml
import pandas as pd
import argparse
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from knowledge_graph import KnowledgeGraphBuilder
from utils.path_manager import PathManager

DEFAULT_CONFIG_FILE = "config/autophagy_treg_config.yaml"
DEFAULT_CSV_FILE = "data/Autophagy_Treg_literature.csv"

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='文献分析和知识图谱构建工具')
    parser.add_argument('-i', '--input', type=str, 
                        help=f'输入CSV文件路径 (默认: {DEFAULT_CSV_FILE})')
    parser.add_argument('-c', '--config', type=str, 
                        help=f'配置文件路径 (默认: {DEFAULT_CONFIG_FILE})')
    parser.add_argument('-o', '--output', type=str, 
                        help='输出目录路径 (默认: 配置文件中指定的输出目录)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细信息')
    return parser.parse_args()

CONFIG_FILE = DEFAULT_CONFIG_FILE
path_manager = None
config_data = None

try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    path_manager = PathManager(CONFIG_FILE)
    print(f"Config file loaded: {CONFIG_FILE}")
except Exception as e:
    print(f"Could not load config file: {e}")
    
    create_config_if_not_exists(CONFIG_FILE)
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        path_manager = PathManager(CONFIG_FILE)
        print(f"Created and loaded default config: {CONFIG_FILE}")
    except Exception as e:
        print(f"Failed to create or load default config: {e}")
        
        path_manager = None
        config_data = None


nltk.data.path.insert(0, "nltk")
print(f"NLTK data path: {nltk.data.path}")


class SimpleAnalyzer:
    """Class for analyzing autophagy and Treg literature"""

    def __init__(self, csv_file=DEFAULT_CSV_FILE, config=None):
        """Initialize analyzer"""
        self.csv_file = csv_file
        self.df = None
        self.lemmatizer = WordNetLemmatizer()
        self.config = config if config else config_data
        
        
        self.stop_words = set(stopwords.words('english'))
        
        self.custom_stopwords = {
            "study", "studies", "result", "results", "method", "methods",
            "figure", "table", "data", "analysis", "analyzed", "et", "al",
            "author", "authors", "journal", "article", "research", "review",
            "investigated", "reported", "showed", "observed", "found", "conclusion",
            "conclusions", "aim", "aims", "objective", "objectives", "background",
            "significance", "were", "was", "been", "being", "have", "has", "had",
            "effect", "effects", "pathway", "pathways", "model", "models"
        }
        self.stop_words.update(self.custom_stopwords)
        
        if self.config and 'entities' in self.config:
            if 'autophagy_terms' in self.config['entities']:
                self.autophagy_terms = set([term.lower() for term in self.config['entities']['autophagy_terms']])
            else:
                self.autophagy_terms = {
                    "autophagy", "autophagic", "autophagosomes", "autolysosome", 
                    "macroautophagy", "mitophagy", "pexophagy", "microautophagy",
                    "atg", "lc3", "beclin", "p62", "sqstm1", "unc-51", "ulk1", "lamp2",
                    "autophagosome", "lysosome", "mtor", "ampk", "rapamycin", "bafilomycin"
                }
            
            if 'treg_terms' in self.config['entities']:
                self.treg_terms = set([term.lower() for term in self.config['entities']['treg_terms']])
            else:
                self.treg_terms = {
                    "treg", "regulatory t", "regulatory t cell", "regulatory t cells", 
                    "foxp3", "cd25", "ctla-4", "gitr", "il-10", "tgf-beta", "pde8a",
                    "t regulatory", "tgf-β", "cd4+cd25+", "immunosuppression", "immunosuppressive",
                    "suppressive", "t-regulatory", "immune tolerance", "self-tolerance"
                }
            
            if 'disease_terms' in self.config['entities']:
                self.disease_terms = set([term.lower() for term in self.config['entities']['disease_terms']])
            else:
                self.disease_terms = {
                    "autoimmune", "inflammation", "cancer", "tumor", "infection", 
                    "allergy", "asthma", "colitis", "arthritis"
                }
        else:
            self.autophagy_terms = {
                "autophagy", "autophagic", "autophagosomes", "autolysosome", 
                "macroautophagy", "mitophagy", "pexophagy", "microautophagy",
                "atg", "lc3", "beclin", "p62", "sqstm1", "unc-51", "ulk1", "lamp2",
                "autophagosome", "lysosome", "mtor", "ampk", "rapamycin", "bafilomycin"
            }
            
            self.treg_terms = {
                "treg", "regulatory t", "regulatory t cell", "regulatory t cells", 
                "foxp3", "cd25", "ctla-4", "gitr", "il-10", "tgf-beta", "pde8a",
                "t regulatory", "tgf-β", "cd4+cd25+", "immunosuppression", "immunosuppressive",
                "suppressive", "t-regulatory", "immune tolerance", "self-tolerance"
            }
            
            self.disease_terms = {
                "autoimmune", "inflammation", "cancer", "tumor", "infection", 
                "allergy", "asthma", "colitis", "arthritis"
            }
        
        if self.config and 'relations' in self.config:
            self.relation_terms = {}
            for rel_type, rel_data in self.config['relations'].items():
                if 'variants' in rel_data:
                    for variant in rel_data['variants']:
                        self.relation_terms[variant.lower()] = rel_type
        else:
            self.interaction_terms = {
                "regulate", "regulates", "regulated", "regulating", "regulation",
                "induce", "induces", "induced", "inducing", "induction",
                "inhibit", "inhibits", "inhibited", "inhibiting", "inhibition",
                "promote", "promotes", "promoted", "promoting", "promotion",
                "suppress", "suppresses", "suppressed", "suppressing", "suppression",
                "activate", "activates", "activated", "activating", "activation",
                "modulate", "modulates", "modulated", "modulating", "modulation"
            }
        
    def load_data(self):
        """Load CSV data"""
        try:
            print(f"Loading literature data: {self.csv_file}")
            self.df = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.df)} publications")
            
            self.analyze_basic_stats()
            
            return True
        except Exception as e:
            print(f"Failed to load data: {e}")
            return False
    
    def analyze_basic_stats(self):
        """Basic statistical analysis"""
        if self.df is None:
            print("Please load data first")
            return
        
        print("\nBasic Information:")
        print(f"- Number of publications: {len(self.df)}")
        
        if 'Year' in self.df.columns:
            years = self.df['Year'].value_counts().sort_index()
            print(f"- Year range: {years.index.min()} - {years.index.max()}")
            print(f"- Year distribution: {dict(years)}")
        
        if 'Journal' in self.df.columns:
            journals = self.df['Journal'].value_counts().head(10)
            print(f"- Main journals (top 10): {dict(journals)}")
    
    def preprocess_text(self, text):
        """Preprocess text"""
        if pd.isna(text):
            return []
        
        text = text.lower()
        
        text = re.sub(r'<.*?>', ' ', text)
        
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        tokens = word_tokenize(text)
        
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def find_entity_in_text(self, text, entity_set):
        """Find entities in text"""
        text = text.lower()
        found_entities = []
        
        for entity in entity_set:
            if entity in text:
                found_entities.append(entity)
                
        return found_entities
    
    def find_relation_in_text(self, text):
        """Find relation terms in text"""
        text = text.lower()
        found_relations = []
        
        if hasattr(self, 'relation_terms'):
            for rel_term, rel_type in self.relation_terms.items():
                if rel_term in text:
                    found_relations.append((rel_term, rel_type))
        else:
            for term in self.interaction_terms:
                if term in text:
                    found_relations.append(term)
                    
        return found_relations
    
    def find_autophagy_treg_connections(self):
        """Find connections between autophagy and regulatory T cells"""
        if self.df is None:
            print("Please load data first")
            return
        
        connections = []
        
        for _, row in self.df.iterrows():
            if pd.isna(row['Abstract']):
                continue
                
            abstract = row['Abstract'].lower()
            
            autophagy_entities = self.find_entity_in_text(abstract, self.autophagy_terms)
            treg_entities = self.find_entity_in_text(abstract, self.treg_terms)
            
            if autophagy_entities and treg_entities:
                # Split into sentences
                sentences = sent_tokenize(abstract)
                for sentence in sentences:
                    sentence_autophagy = self.find_entity_in_text(sentence, self.autophagy_terms)
                    sentence_treg = self.find_entity_in_text(sentence, self.treg_terms)
                    relations = self.find_relation_in_text(sentence)
                    
                    if sentence_autophagy and sentence_treg and relations:
                        # Found a sentence containing autophagy, Treg and relation terms
                        connection_info = {
                            'PMID': row['PMID'],
                            'Title': row['Title'],
                            'Year': row['Year'],
                            'Sentence': sentence.strip(),
                            'Autophagy_terms': sentence_autophagy,
                            'Treg_terms': sentence_treg,
                            'Relations': relations
                        }
                        connections.append(connection_info)
        
        
        connections_df = pd.DataFrame(connections)
        
        if not connections_df.empty:
            print(f"\nFound {len(connections_df)} connections between autophagy and Treg")
            connections_df.to_csv("autophagy_treg_connections.csv", index=False)
            print("Connections saved to: autophagy_treg_connections.csv")
            
            
            relation_counts = Counter()
            for _, row in connections_df.iterrows():
                if isinstance(row['Relations'], list):
                    for rel in row['Relations']:
                        if isinstance(rel, tuple):
                            relation_counts[rel[1]] += 1  # Count relation types
                        else:
                            relation_counts[rel] += 1
            
            print("\nRelation type distribution:")
            for rel, count in relation_counts.most_common():
                print(f"- {rel}: {count}")
        else:
            print("\nNo direct interactions found between autophagy and Treg")
        
        return connections_df
    
    def prepare_data_for_knowledge_graph(self):
        """Prepare data for knowledge graph construction"""
        if self.df is None:
            print("Please load data first")
            return None
        
        kg_data = self.df.copy()
        
        output_file = "data/analyzed_literature.csv"
        kg_data.to_csv(output_file, index=False)
        print(f"Knowledge graph input data saved to: {output_file}")
        
        return output_file
    
    def build_knowledge_graph(self):
        """Build knowledge graph
        
        This method now uses a dictionary-based approach instead of the biolinkbert model,
        which makes entity extraction more reliable but potentially less accurate.
        """
        global CONFIG_FILE
        
        data_file = self.prepare_data_for_knowledge_graph()
        if data_file is None:
            return
        
        print("\nBuilding knowledge graph using dictionary-based entity extraction...")
        builder = KnowledgeGraphBuilder(CONFIG_FILE)
        builder.build_graph(data_file)
        
        if len(builder.G.nodes) > 0:
            print("\nKnowledge graph construction completed!")
            print(f"- Number of nodes: {len(builder.G.nodes)}")
            print(f"- Number of edges: {len(builder.G.edges)}")

            visualize = input("Do you want to generate interactive visualizations? (y/n): ").lower().strip()
            if visualize == 'y':
                builder.visualize_graph()
                
                builder.export_for_echarts()
            
            builder.export_graph()
            
        else:
            print("\nFailed to build the graph, no files generated")


def create_config_if_not_exists(config_file=DEFAULT_CONFIG_FILE):
    """Create default configuration if it doesn't exist"""
    config_dir = os.path.dirname(config_file)
    os.makedirs(config_dir, exist_ok=True)
    
    if not os.path.exists(config_file):
        print(f"Config file not found, creating default config: {config_file}")
        
        # Create default configuration content
        config_content = """
# Autophagy and Regulatory T Cell Literature Analysis Configuration
domain: "autophagy-treg"

paths:
  logs: "logs"
  output: "output"
  analysis:
    wordcloud: "output/wordcloud"
    trends: "output/trends"
  knowledge_graph: "output/knowledge_graph"

file_names:
  data: "data/Autophagy_Treg_literature.csv"
  analysis: "data/analyzed_literature.csv"

entities:
  autophagy_terms:
    - autophagy
    - autophagic
    - autophagosome
    - autolysosome
    - mitophagy
    - LC3
    - Beclin
    - ATG
    - SQSTM1/p62
    - mTOR
    - ULK1
    - AMPK

  treg_terms:
    - regulatory T cell
    - Treg
    - FOXP3
    - CD4+CD25+
    - IL-10
    - TGF-beta
    - immunosuppression
    - immune tolerance

  disease_terms:
    - autoimmune
    - inflammation
    - cancer
    - tumor
    - infection
    - allergy
    - asthma
    - colitis
    - arthritis

relations:
  activates:
    variants: [activates, activate, activating, activated, activation, induces, induce, inducing, induced, induction]
    type: "positive_regulation"
    
  inhibits:
    variants: [inhibits, inhibit, inhibiting, inhibited, inhibition, suppresses, suppress, suppressing, suppressed, suppression]
    type: "negative_regulation"
    
  regulates:
    variants: [regulates, regulate, regulating, regulated, regulation, controls, control, controlling, controlled]
    type: "regulation"

performance:
  entity_recognition:
    max_term_length: 25
    nested_entity_handling: merge

visualization:
  node_size_formula: "log(weight) * 10"
  edge_width_range: [1, 5]
  
  color_scheme:
    entity_types:
      GENE: "#FF7F0E"       # 橙色
      CHEMICAL: "#2CA02C"   # 绿色
      DISEASE: "#D62728"    # 红色
      CELL_TYPE: "#1F77B4"  # 蓝色
      DEFAULT: "#9467BD"    # 紫色 (默认/未识别)
    
    relation_types:
      positive_regulation: "#00a300"  # 深绿色 - 激活/诱导关系
      negative_regulation: "#cc0000"  # 深红色 - 抑制关系
      regulation: "#0066cc"           # 蓝色 - 一般调控关系
      DEFAULT: "#999999"              # 灰色 - 默认/未分类关系
      
  styles:
    node:
      border_width: 2
      border_width_selected: 3
      shape: "dot"  # 可选: dot, ellipse, triangle, square 等
    edge:
      smooth: false
      arrows: false
      dashes: false
"""
        
        # Write to config file
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_content)
        
        return True
    
    return False


def setup_environment(args):
    """设置运行环境"""
    global path_manager, config_data, CONFIG_FILE
    
    if args.config:
        CONFIG_FILE = args.config
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        path_manager = PathManager(CONFIG_FILE)
        print(f"配置文件已加载: {CONFIG_FILE}")
    except Exception as e:
        print(f"无法加载配置文件: {e}")
        
        create_config_if_not_exists(CONFIG_FILE)
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            path_manager = PathManager(CONFIG_FILE)
            print(f"已创建并加载默认配置: {CONFIG_FILE}")
        except Exception as e:
            print(f"无法创建或加载默认配置: {e}")
            
            path_manager = None
            config_data = None
    
    print(f"NLTK数据路径: {nltk.data.path}")
    
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if args.output and config_data and 'paths' in config_data:
        config_data['paths']['output'] = args.output
        
        if 'analysis' in config_data['paths']:
            config_data['paths']['analysis']['wordcloud'] = os.path.join(args.output, "wordcloud")
            config_data['paths']['analysis']['trends'] = os.path.join(args.output, "trends")
        config_data['paths']['knowledge_graph'] = os.path.join(args.output, "knowledge_graph")
        
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(os.path.join(args.output, "wordcloud"), exist_ok=True)
        os.makedirs(os.path.join(args.output, "trends"), exist_ok=True)
        os.makedirs(os.path.join(args.output, "knowledge_graph"), exist_ok=True)


def main():
    """主函数"""
    args = parse_arguments()
    
    setup_environment(args)
    
    print("=" * 50)
    print("Simple Autophagy and Regulatory T Cell Literature Analysis")
    print("=" * 50)
    print("Note: Using dictionary-based entity extraction instead of biolinkbert model")
    print("=" * 50)
    
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    csv_file = args.input if args.input else DEFAULT_CSV_FILE
    print(f"使用输入文件: {csv_file}")
    
    analyzer = SimpleAnalyzer(csv_file=csv_file, config=config_data)
    
    if analyzer.load_data():
        analyzer.find_autophagy_treg_connections()
        
        build_graph = input("\n是否构建知识图谱? (y/n): ").lower().strip()
        if build_graph == 'y':
            analyzer.build_knowledge_graph()
    
    print("\n分析完成!")


if __name__ == "__main__":
    main() 