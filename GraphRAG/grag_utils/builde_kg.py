"""
将结构化的三元组json文件存入 Neo4j 数据库中。
"""

from dotenv import load_dotenv
from pathlib import Path
import os
import json
from neo4j import GraphDatabase


# ---------- 环境导入 ----------
project_root = Path(__file__).parent.parent.parent  # 项目根目录
env_path = project_root / "asst.env"  # 环境变量文件路径
load_dotenv(env_path)
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")


class Neo4jImporter:
    """
    Neo4j数据库导入器
    """
    
    def __init__(self, uri="bolt://localhost:7687", user=neo4j_username, password=neo4j_password):
        """
        初始化Neo4j连接
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """
        关闭数据库连接
        """
        self.driver.close()
    
    def import_entities(self, entities_file):
        """
        导入实体数据到Neo4j
        """
        with self.driver.session() as session:
            # 读取实体文件
            with open(entities_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_entities = len(lines)
            print(f"开始导入 {total_entities} 个实体...")
            
            for i, line in enumerate(lines, 1):
                try:
                    entity_data = json.loads(line.strip())
                    entity_name = entity_data['name']
                    properties = entity_data.get('properties', {})
                    relationships = entity_data.get('relationships', [])
                    
                    # 执行Cypher语句创建节点和关系
                    session.execute_write(
                        self._create_entity,
                        entity_name, properties, relationships
                    )
                    
                    if i % 100 == 0:
                        print(f"已导入 {i}/{total_entities} 个实体")
                        
                except Exception as e:
                    print(f"导入第 {i} 个实体时出错: {e}")
            
            print(f"导入完成！共导入 {total_entities} 个实体")
    
    @staticmethod
    def _create_entity(tx, entity_name, properties, relationships):
        """
        创建单个实体及其关系的Cypher语句
        """
        # 创建实体节点并设置属性
        # 构建SET语句
        set_clauses = []
        params = {'entity_name': entity_name}
        
        for key, value in properties.items():
            param_name = f'prop_{key}'
            params[param_name] = value
            set_clauses.append(f"a.{key} = ${param_name}")  # 动态构造
        
        set_statement = " SET " + ", ".join(set_clauses) if set_clauses else ""
        
        # 创建实体节点
        tx.run(f"""
        MERGE (a:Entity {{name: $entity_name}})
        {set_statement}
        """, **params)
        
        # 创建关系
        for rel in relationships:
            rel_type = rel['type']
            target = rel['target']
            
            tx.run("""
            MERGE (a:Entity {name: $entity_name})
            MERGE (b:Entity {name: $target})
            MERGE (a)-[r:RELATION {type: $rel_type}]->(b)
            """, entity_name=entity_name, target=target, rel_type=rel_type)


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
    importer = Neo4jImporter()
    
    try:
        # 导入数据
        importer.import_entities(entities_file)
    finally:
        # 关闭连接
        importer.close()


if __name__ == "__main__":
    main()