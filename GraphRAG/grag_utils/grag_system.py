"""
GraphRAG 系统

功能说明：
- 基于 Milvus 向量数据库进行语义检索（向量召回阶段）
- 使用 OpenAI-compatible Embedding API 对查询进行向量化
- 从检索结果中抽取 subject 作为图入口节点
- 使用 Neo4j 图数据库扩展子图，获取相关实体和关系
- 最终返回结构化的子图信息供上层 LLM 生成回答

系统架构说明：
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
结构化子图信息
"""

from dotenv import load_dotenv
from pathlib import Path
import os
from openai import OpenAI
from pymilvus import MilvusClient
from neo4j import GraphDatabase


# ----------------------------------------
# 导入环境与配置文件
# ----------------------------------------
project_root = Path(__file__).parent.parent.parent     # 项目根目录
env_path = project_root / "asst.env"                   # 环境变量文件路径
load_dotenv(env_path)
openai_api_key = os.getenv("OPENAI_API_KEY")           # OpenAI API Key
api_base_url = os.getenv("API_BASE_URL")               # API Base URL
milvus_collection = os.getenv("MILVUS_COLLECTION")     # Milvus集合名称
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")  # Neo4j用户名
neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")  # Neo4j密码


# =================================================
# GraphRAGSystem：负责向量召回 + 图子图扩展
# =================================================
class GraphRAGSystem:
    """
    GraphRAG 核心系统类
    
    负责：
    1. 初始化 Milvus、Neo4j 和 Embedding 客户端
    2. 处理用户查询，执行向量召回
    3. 从召回结果中提取入口节点
    4. 基于入口节点扩展子图
    5. 生成 LLM 友好的上下文信息
    """
    
    def __init__(
        self,
        milvus_uri="http://localhost:19530",
        collection_name="harry_potter",  # Milvus集合名称
        neo4j_uri="bolt://localhost:7687",  # Neo4j连接URI
    ):
        # 初始化 Milvus 客户端
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name
        
        # 初始化 Neo4j 客户端
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password)
        )
        
        # 初始化 OpenAI Embedding 客户端
        self.embedding_client = OpenAI(api_key=openai_api_key, base_url=api_base_url)
        
        # 轻量化优化参数
        self.max_entry_nodes = 3  # 最大入口节点数量
        self.max_entities = 5  # 最大实体数量
        self.max_relations_per_entity = 3  # 每个实体的最大关系数量
        self.max_triples = 5  # 最大三元组数量
        self.max_context_length = 1000  # 最大上下文长度
        
        # 输出初始化信息
        print(f"GraphRAGSystem 初始化完成")
        print(f"  Milvus: {milvus_uri}, 集合: {collection_name}")
        print(f"  Neo4j: {neo4j_uri}")
        print(f"  轻量化参数: 最大入口节点={self.max_entry_nodes}, 最大实体={self.max_entities}")
    
    def close(self):
        """
        关闭数据库连接
        """
        self.neo4j_driver.close()
        print("GraphRAGSystem 连接已关闭")
    
    def retrieval_chunks(self, query, n_results=5, hop_distance=2):
        """
        检索相关子图信息，返回LLM友好的上下文
        
        参数:
            query: 用户查询文本
            n_results: Milvus召回数量
            hop_distance: Neo4j跳数
        
        返回:
            {
                "documents": ["相关文本1", "相关文本2", ...],  # LLM友好的上下文
                "metadatas": [{}, {}, ...]  # 元数据
            }
        
        逻辑流程:
        1. 向量化用户查询
        2. Milvus 向量召回相似三元组
        3. 提取 subject 作为图入口节点
        4. Neo4j 扩展子图
        5. 生成 LLM 友好的上下文
        """
        # 输出检索开始信息
        print(f"\n{'='*60}")
        print(f"🔍 开始 GraphRAG 检索")
        print(f"{'='*60}")
        print(f"查询问题: {query}")
        print(f"Milvus召回数量: {n_results}")
        print(f"Neo4j跳数: {hop_distance}")
        print(f"{'='*60}\n")
        
        # 1. 向量化用户提问
        print("📝 步骤1: 向量化用户问题...")
        response = self.embedding_client.embeddings.create(
            model="text-embedding-v4",
            input=[query],
            dimensions=1024,
        )
        query_vector = response.data[0].embedding
        print(f"✅ 向量化完成，向量维度: {len(query_vector)}")
        
        # 2. Milvus 向量召回（检索相似三元组）
        print(f"\n📊 步骤2: Milvus 向量召回...")
        result = self.milvus_client.search(
            collection_name=self.collection_name,
            anns_field="vector",
            data=[query_vector],
            limit=n_results,
            output_fields=["subject", "predicate", "object", "content"],
            search_params={
                "metric_type": "COSINE",
                "params": {
                    "nprobe": 10,  # 搜索桶数量
                    "radius": 0.5,  # 余弦距离阈值
                },
            },
        )
        
        # 3. 提取检索结果
        triples = []  # 存储三元组信息
        subjects = set()  # 存储入口节点
        total_relationships = 0  # 总关系数
        
        if len(result) and len(result[0]):
            print(f"✅ Milvus 检索到 {len(result[0])} 个相似三元组:\n")
            for i, item in enumerate(result[0], 1):
                distance = item['distance']
                entity = item['entity']
                triple = {
                    'subject': entity['subject'],
                    'predicate': entity['predicate'],
                    'object': entity['object'],
                    'content': entity['content'],
                    'score': 1 - distance  # 余弦相似度 = 1 - 余弦距离
                }
                triples.append(triple)
                subjects.add(entity['subject'])
                
                # 输出三元组信息
                print(f"  [{i}] 相似度: {triple['score']:.4f} | {triple['content']}")
                
                # 限制三元组数量
                if len(triples) >= self.max_triples:
                    break
        else:
            print("⚠️  Milvus 未检索到相关三元组")
            # 返回空结果，触发RAG回退
            return {
                "documents": [],
                "metadatas": []
            }
        
        # 4. 提取入口节点（限制数量）
        entry_nodes = list(subjects)[:self.max_entry_nodes]
        print(f"\n🎯 步骤3: 抽取图入口节点...")
        print(f"✅ 提取到 {len(entry_nodes)} 个图入口节点: (限制为{self.max_entry_nodes}个)")
        for i, node in enumerate(entry_nodes, 1):
            print(f"  [{i}] {node}")
        
        # 5. Neo4j 子图扩展
        print(f"\n🕸️  步骤4: Neo4j 子图扩展...")
        
        subgraph = {}  # 存储子图信息
        with self.neo4j_driver.session() as session:
            for node_name in entry_nodes:
                print(f"\n  📍 扩展节点: {node_name}")
                
                # ------------------------------
                # 查询子图
                # 1. 找到入口节点
                # 2. 扩展跳数：1~hop_distance跳  (n)-[r*1..2]-(related)
                #     2跳原因：多跳会爆炸（节点数量、子图、LLM上下文）
                # 3. 收集路径：把路径里的每一条关系类型提取出来 
                # ------------------------------
                # 构建轻量级Cypher查询
                # 限制返回的关系数量，减少数据传输
                query_cypher = f"""
                MATCH (n:Entity {{name: $node_name}})
                OPTIONAL MATCH (n)-[r*1..{hop_distance}]-(related)
                WHERE size(collect(r)) <= 10  // 限制关系数量
                WITH n, collect(DISTINCT {{relation: [rel in r | type(rel)], target: related.name}})[0..5] as paths
                RETURN n.name as name, n.性别 as gender, n.物种 as species, n.出生 as birth, paths
                """
                
                result = session.run(query_cypher, node_name=node_name)
                
                for record in result:
                    entity_name = record["name"]
                    paths = record["paths"]
                    
                    # 构建实体信息
                    entity_info = {
                        "name": entity_name,
                        "properties": {
                            "gender": record["gender"],
                            "species": record["species"],
                            "birth": record["birth"]
                        },
                        "relationships": []
                    }
                    
                    # 解析关系路径（限制数量）
                    relation_count = 0
                    for path in paths:
                        if path["relation"] and path["target"] and relation_count < self.max_relations_per_entity:
                            relations = path["relation"]
                            target = path["target"]
                            for relation in relations:
                                entity_info["relationships"].append({
                                    "type": relation,
                                    "target": target
                                })
                                relation_count += 1
                                if relation_count >= self.max_relations_per_entity:
                                    break
                        if relation_count >= self.max_relations_per_entity:
                            break
                    
                    subgraph[entity_name] = entity_info
                    total_relationships += len(entity_info["relationships"])
                    
                    # 输出关系数量
                    print(f"    ✅ 找到 {len(entity_info['relationships'])} 个关系 (限制为{self.max_relations_per_entity}个)")
                    
                    # 限制实体数量
                    if len(subgraph) >= self.max_entities:
                        break
                
                # 限制实体数量
                if len(subgraph) >= self.max_entities:
                    break
        
        # 6. 格式化LLM上下文
        print(f"\n📋 步骤5: 整理子图信息...")
        
        context_parts = []  # 上下文部分
        
        # 添加三元组信息（限制数量）
        if triples:
            context_parts.append("相关三元组:")
            for triple in triples[:self.max_triples]:
                context_parts.append(f"  - {triple['content']}")
        
        # 添加子图信息（限制数量）
        if subgraph:
            context_parts.append("\n相关实体信息:")
            for entity_name, entity_info in list(subgraph.items())[:self.max_entities]:
                props = entity_info["properties"]
                rels = entity_info["relationships"]
                
                # 构建实体描述
                entity_desc = f"  {entity_name}"
                if props["gender"]:
                    entity_desc += f" ({props['gender']})"
                if props["species"]:
                    entity_desc += f", {props['species']}"
                
                # 添加关系信息（限制数量）
                if rels:
                    entity_desc += " - 关系: "
                    rel_descs = [f"{rel['type']}:{rel['target']}" for rel in rels[:self.max_relations_per_entity]]
                    entity_desc += ", ".join(rel_descs)
                
                context_parts.append(entity_desc)
        
        # 组合上下文
        context_text = "\n".join(context_parts)
        
        # 限制上下文长度
        if len(context_text) > self.max_context_length:
            context_text = context_text[:self.max_context_length] + "..."
        
        # 输出完成信息
        print(f"✅ 子图扩展完成，共 {len(subgraph)} 个实体，{total_relationships} 个关系")
        print(f"✅ 上下文长度: {len(context_text)} 字符 (限制为{self.max_context_length})")
        print(f"{'='*60}\n")
        
        # 返回RAG兼容的格式
        return {
            "documents": [context_text],  # LLM友好的上下文
            "metadatas": [{
                "source": "GraphRAG",  # 来源标识
                "entities": list(subgraph.keys())  # 涉及的实体列表
            }]
        }


# =================================================
# GraphRAGService：单例模式服务类
# =================================================
class GraphRAGService:
    """
    GraphRAG 服务类
    
    使用单例模式，确保全局只有一个 GraphRAGSystem 实例
    提供简洁的接口调用方法
    """
    
    _instance = None  # 单例实例
    
    @classmethod
    def get_instance(cls):
        """
        获取 GraphRAGSystem 实例
        
        返回:
            GraphRAGSystem 实例
        """
        if cls._instance is None:
            cls._instance = GraphRAGSystem()
        return cls._instance
    
    @classmethod
    def retrieval_chunks(cls, query, n_results=5, hop_distance=2):
        """
        检索相关子图信息
        
        参数:
            query: 用户查询文本
            n_results: Milvus召回数量
            hop_distance: Neo4j跳数
        
        返回:
            {
                "documents": ["相关文本1", "相关文本2", ...],  # LLM友好的上下文
                "metadatas": [{}, {}, ...]  # 元数据
            }
        """
        grag_system = cls.get_instance()
        return grag_system.retrieval_chunks(
            query=query,
            n_results=n_results,
            hop_distance=hop_distance
        )


# =================================================
# 测试函数
# =================================================
def main():
    """
    测试 GraphRAG 系统
    """
    print("🚀 GraphRAG 系统测试 (轻量化版本)\n")
    
    # 测试查询
    test_queries = [
        "哈利·波特的家庭成员有哪些？",
        "邓布利多的职业是什么？",
        "霍格沃茨的四位创始人是谁？"
    ]
    
    for query in test_queries:
        print(f"\n{'#'*60}")
        print(f"# 测试查询: {query}")
        print(f"{'#'*60}\n")
        
        # 使用服务层调用
        result = GraphRAGService.retrieval_chunks(
            query=query,
            n_results=5,
            hop_distance=2
        )
        
        # 输出结果
        print(f"\n📝 LLM 上下文:\n{result['documents'][0] if result['documents'] else '无结果'}\n")
        print(f"{'#'*60}\n")
        
        # 暂停一下，避免输出太快
        import time
        time.sleep(1)
    
    print("🎯 测试完成！")


if __name__ == "__main__":
    main()
