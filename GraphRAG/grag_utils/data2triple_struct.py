import json
import os

def convert_json_to_triples(input_file, output_file):
    """
    将半结构化的json数据转换为三元组的形式
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entities = []
    
    # 遍历每个人物
    for character, properties in data.items():
        # 创建人物实体
        entity = {
            'type': 'entity',
            'name': character,
            'properties': {}
        }
        
        # 添加基本属性
        basic_properties = ['出生', '性别', '物种']
        for prop in basic_properties:
            if prop in properties:
                entity['properties'][prop] = properties[prop]
        
        # 添加关系
        entity['relationships'] = []
        
        # 处理家庭信息
        if '家庭信息' in properties:
            family_info = properties['家庭信息']
            for relation, value in family_info.items():
                if isinstance(value, list):
                    # 处理数组类型的值
                    for item in value:
                        entity['relationships'].append({
                            'type': relation,
                            'target': item
                        })
                else:
                    # 处理单个值
                    entity['relationships'].append({
                        'type': relation,
                        'target': value
                    })
        
        # 处理职业
        if '职业' in properties:
            professions = properties['职业']
            if isinstance(professions, list):
                for profession in professions:
                    entity['relationships'].append({
                        'type': '职业',
                        'target': profession
                    })
            else:
                entity['relationships'].append({
                    'type': '职业',
                    'target': professions
                })
        
        # 处理从属
        if '从属' in properties:
            affiliations = properties['从属']
            if isinstance(affiliations, list):
                for affiliation in affiliations:
                    entity['relationships'].append({
                        'type': '从属',
                        'target': affiliation
                    })
            else:
                entity['relationships'].append({
                    'type': '从属',
                    'target': affiliations
                })
        
        # 处理学院
        if '学院' in properties:
            academy = properties['学院']
            entity['relationships'].append({
                'type': '学院',
                'target': academy
            })
        
        entities.append(entity)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entity in entities:
            json.dump(entity, f, ensure_ascii=False)
            f.write('\n')

if __name__ == "__main__":
    # 定义文件路径
    input_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'harry_potter_property.json')
    output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'harry_potter.json')
    
    # 执行转换
    convert_json_to_triples(input_file, output_file)
    print(f"转换完成，输出文件: {output_file}")
