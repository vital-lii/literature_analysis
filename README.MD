## 文献可视化程序
简写：
    S_A: Simple Analysis
    K_G: Knowledge Graph
    W_C: Word Cloud
    T_S: Trends
SA.PY为运行程序，但形成GEXF文件，需调用KG.PY。二者绑定。
config文件夹下为配置文件，可以修改，但尽量不要修改文件名，避免程序运行错误。
utils文件夹下为工具函数不要动。
NLTK文件夹下为nltk的文件，不要动，断词汇用。

### 文件输出
包含： 
    GEXF文件：知识图谱
    CSV文件：分析结果，共两个文件，一个为分析结果包含出现关系词，一个为中间文件传递到KG.PY进行分析生成GEXF文件。
    


