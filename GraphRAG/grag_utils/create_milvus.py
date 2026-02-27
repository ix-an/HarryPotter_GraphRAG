"""
创建 Milvus 索引
1.读取 harry_potter.json（实体格式）
2.把每个实体转换成自然语言文本
3.embedding 并存入 Milvus
"""

from dotenv import load_dotenv
from pathlib import Path
import os
import json
from openai import OpenAI
from pymilvus import MilvusClient


project_root = Path(__file__).parent.parent.parent  # 项目根目录
env_path = project_root / "asst.env"  # 环境变量文件路径
load_dotenv(env_path)
openai_api_key = os.getenv("OPENAI_API_KEY")  
api_base_url = os.getenv("API_BASE_URL")


class MilvusImporter:
    """
    Milvus 数据导入器
    """
    
    def __init__(self, uri="http://localhost:19530", collection_name="harry_potter"):
        """
        初始化
        """
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.embedding_client = OpenAI(api_key=openai_api_key, base_url=api_base_url)
    
    def create_collection(self):
        """
        创建 Milvus 集合
        """
        # 检查集合是否存在
        if self.client.has_collection(collection_name=self.collection_name):
            print(f"集合 {self.collection_name} 已存在，删除后重建")
            self.client.drop_collection(collection_name=self.collection_name)
        
        # 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=1024,  # 向量维度
            auto_id=True,
            primary_field="id",
            vector_field="vector",
            metric_type="COSINE"  # 余弦相似度
        )
        print(f"集合 {self.collection_name} 创建成功")
    
    def generate_triples(self, entity):
        """
        生成三元组
        """
        triples = []
        name = entity['name']
        relationships = entity.get('relationships', [])
        
        # 处理关系
        for rel in relationships:
            triple_text = f"{name} 的 {rel['type']} 是 {rel['target']}"
            triples.append({
                "subject": name,
                "predicate": rel['type'],
                "object": rel['target'],
                "text": triple_text
            })
        
        return triples
    
    def embedding_text(self, text):
        """
        文本向量化
        """
        response = self.embedding_client.embeddings.create(
            model="text-embedding-v4",
            input=[text],
            dimensions=1024,
        )
        return response.data[0].embedding
    
    def import_data(self, entities_file):
        """
        导入数据到 Milvus
        """
        # 创建集合
        self.create_collection()
        
        # 读取实体文件
        with open(entities_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_entities = len(lines)
        print(f"开始处理 {total_entities} 个实体...")
        
        batch_size = 100
        batch = []
        total_triples = 0
        
        for i, line in enumerate(lines, 1):
            try:
                entity = json.loads(line.strip())
                
                # 生成三元组
                triples = self.generate_triples(entity)
                total_triples += len(triples)
                
                # 处理每个三元组
                for triple in triples:
                    # 向量化
                    vector = self.embedding_text(triple['text'])
                    
                    # 构建数据
                    data = {
                        "vector": vector,
                        "content": triple['text'],
                        "subject": triple['subject'],  # 显式添加subject字段
                        "predicate": triple['predicate'],
                        "object": triple['object'],
                        "metadata": {
                            "subject": triple['subject'],
                            "predicate": triple['predicate'],
                            "object": triple['object']
                        }
                    }
                    
                    batch.append(data)
                    
                    # 批量插入
                    if len(batch) >= batch_size:
                        self.client.insert(
                            collection_name=self.collection_name,
                            data=batch
                        )
                        print(f"已处理 {i}/{total_entities} 个实体，导入 {len(batch)} 个三元组")
                        batch = []
                        
            except Exception as e:
                print(f"处理第 {i} 个实体时出错: {e}")
        
        # 插入剩余数据
        if batch:
            self.client.insert(
                collection_name=self.collection_name,
                data=batch
            )
            print(f"已处理 {total_entities}/{total_entities} 个实体，导入 {len(batch)} 个三元组")
        
        print(f"导入完成！共处理 {total_entities} 个实体，导入 {total_triples} 个三元组")


def main():
    """
    主函数
    """
    # 定义文件路径
    graphrag_dir = Path(__file__).parent.parent
    entities_file = graphrag_dir / "data" / "harry_potter.json"
    
    if not entities_file.exists():
        print(f"文件不存在: {entities_file}")
        return
    
    # 创建导入器实例
    importer = MilvusImporter()
    
    # 导入数据
    importer.import_data(entities_file)


if __name__ == "__main__":
    main()