"""
将抽取出的三元组json文件转换为RDF文件，并保存。
RDF（Resource Description Framework）是一种用于描述资源的元数据模型。
它由三部分组成：资源（Resource）、属性（Property）和值（Value）。
每个资源都有一个唯一的标识符（URI），属性描述了资源的特征，值则是属性的具体取值。
"""

import json
import re

with open('harry_potter_property.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

character_set = set()
group_set = set()
house_set = set()
blood_set = set()

for character in data.keys():
    character_set.add(character)
    character_info = data[character]
    if "学院" in character_info:
        house = character_info["学院"]
        house_set.add(house)
    if "从属" in character_info:
        group_list = character_info["从属"]
        for group in group_list:
            group_set.add(group)
    if "血统" in character_info:
        blood = character_info["血统"]
        blood_set.add(blood)

rdf_file = open('harry_potter.rdf', 'w', encoding='utf-8')

rdf_file.write("@prefix relation: <http://kg.course/harry_potter/关系#> .\n")
rdf_file.write("@prefix character: <http://kg.course/harry_potter/角色#> .\n")
rdf_file.write("@prefix group: <http://kg.course/harry_potter/组织#> .\n")
rdf_file.write("@prefix house: <http://kg.course/harry_potter/学院#> .\n")
rdf_file.write("@prefix blood: <http://kg.course/harry_potter/血统#> .\n")
rdf_file.write("\n")


for group in group_set:
    rdf_file.write("group:{}\n\trelation:name \"{}\" ;\n\ta relation:Group .\n".format(group.replace(".", ""), group))

for house in house_set:
    rdf_file.write("house:{}\n\trelation:name \"{}\" ;\n\ta relation:House .\n".format(house, house))

for blood in blood_set:
    rdf_file.write("blood:{}\n\trelation:name \"{}\" ;\n\ta relation:Blood .\n".format(blood, blood))

for character in character_set:
    character_info = data[character]
    character_value = re.sub("[() ]", "", character)
    rdf_file.write("character:{}\n\trelation:name \"{}\" ".format(character_value, character))
    rdf_file.write(";\n\ta relation:Character ")
    for i, key in enumerate(character_info.keys()):
        if key == "家庭信息":
            family_info = character_info["家庭信息"]
            for family_relation in family_info.keys():
                if type(family_info[family_relation]) == list:
                    for name in family_info[family_relation]:
                        if name in character_set:
                            family_relation = family_relation.replace(" ", "，")
                            family_relation = family_relation.replace("/", "或")
                            rdf_file.write(";\n\trelation:{} character:{} ".format(family_relation, name))
                else:
                    name = family_info[family_relation]
                    if name in character_set:
                        family_relation = family_relation.replace(" ", "，")
                        family_relation = family_relation.replace("/", "或")
                        rdf_file.write(";\n\trelation:{} character:{} ".format(family_relation, name))
        elif key == "学院":
            house = character_info["学院"]
            rdf_file.write(";\n\trelation:学院 house:{} ".format(house))
        elif key == "血统":
            blood = character_info["血统"]
            rdf_file.write(";\n\trelation:血统 blood:{} ".format(blood))
        elif key == "从属":
            group_list = character_info["从属"]
            for group in group_list:
                rdf_file.write(";\n\trelation:从属 group:{} ".format(group.replace(".", "")))
        elif key == "职业":
            career_list = character_info["职业"]
            for career in career_list:
                rdf_file.write(";\n\trelation:职业 \"{}\" ".format(career))
        else:
            rdf_file.write(";\n\trelation:{} \"{}\" ".format(key, character_info[key]))
    rdf_file.write(".\n")
