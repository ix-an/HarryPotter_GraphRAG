# HarryPotter_GraphRAG
基于Diet_Assistant项目框架做的知识图谱项目

数据：
  半结构化文档：http://data.openkg.cn/dataset/openkg-harry-potter
  纯文本：《哈利波特与魔法石》前六章内容

整体流程：
原始JSON数据
      ↓
Schema设计（概念建模）
      ↓
三元组抽取规则设计
      ↓
结构化转换（data2triple_struct.py）
      ↓
导入 Neo4j 构建图谱
      ↓
GraphRAG 查询增强

GraphRAG逻辑：
Query (用户问题)
    ↓
Embedding (向量化)
    ↓
Milvus 向量召回 (相似三元组)
    ↓
抽取 Subject (图入口节点)
    ↓
Neo4j 子图扩展 (相关实体和关系)
    ↓
子图裁剪
    ↓
结构化文本构造
    ↓
输入 LLM
