import os
import yaml

class PathManager:
    """路径管理器"""
    
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file  # 保存配置文件路径作为属性
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
    def ensure_dirs(self):
        """确保所有必要的目录存在"""
        dirs = [
            self.config['paths']['logs'],
            self.config['paths']['output'],
            self.config['paths']['analysis']['wordcloud'],
            self.config['paths']['analysis']['trends'],
            self.config['paths']['knowledge_graph']
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            
    def get_path(self, *keys):
        """获取配置的路径"""
        value = self.config['paths']
        for key in keys:
            value = value[key]
        return value
        
    def get_file_path(self, *keys):
        """获取文件路径"""
        value = self.config['file_names']
        for key in keys:
            value = value[key]
        return value 