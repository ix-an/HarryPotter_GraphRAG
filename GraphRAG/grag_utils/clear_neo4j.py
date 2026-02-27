"""
清空 Neo4j 数据库中的所有节点和关系。
默认连接本地 Neo4j 数据库，用户名和密码从环境变量中获取。
"""


import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ----- 加载环境变量 -----
project_root = Path(__file__).parent.parent.parent  # 项目根目录
env_path = project_root / "asst.env"  # 环境变量文件路径
load_dotenv(env_path)
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")

def clear_neo4j_database():
    # 使用环境变量或默认值
    uri = "bolt://localhost:7687"
    username = neo4j_username
    password = neo4j_password
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        with driver.session() as session:
            # 清空数据库中的所有节点和关系
            session.run("MATCH (n) DETACH DELETE n;")
            print("数据库已成功清空")
    except Exception as e:
        print(f"清空数据库时出错: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    clear_neo4j_database()
    # 测试环境变量
    # print("neo4j_passsword:", neo4j_password)
