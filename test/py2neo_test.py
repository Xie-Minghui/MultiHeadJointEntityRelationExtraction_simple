# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/4/17 15:42
# @File    : py2neo_test.py

"""
file description:：

"""
from py2neo import Graph, Node, Relationship, NodeMatcher

test_graph = Graph("http://localhost:7474", username="neo4j", password="root")

# def build_nodes(node_record):
#     data = {"id": str(node_record.n._id), "label": next(iter(node_record.n.labels))}
#     data.update(node_record.n.properties)
#
#     return {"data": data}
#
#
# def build_edges(relation_record):
#     data = {"source": str(relation_record.r.start_node._id),
#             "target": str(relation_record.r.end_node._id),
#             "relationship": relation_record.r.rel.type}
#
#     return {"data": data}

# 清空所有数据对象
# test_graph.delete_all()
# test_graph.schema.create_uniqueness_constraint('Text', 'name')

def build_graph(rel_triple_list):
    matcher = NodeMatcher(test_graph)
    for rel_triple in rel_triple_list:
        if rel_triple is not None:
            for item in rel_triple:
                # print(item)
                node_s = list(matcher.match(item[0], name=item[0]))
                node_o = list(matcher.match(item[1], name=item[1]))
                
                if len(node_s) <= 0:
                    subject = Node(item[0], label=item[3], name=item[0])
                    test_graph.create(subject)
                else:
                    subject = node_s[0]
                if len(node_o) <= 0:
                    object = Node(item[1], label=item[4], name=item[1])
                    test_graph.create(object)
                else:
                    object = node_o[0]
                   
                rel = Relationship(subject, item[2], object)
                rel['type'] = item[2]
                test_graph.create(rel)

#创建节点
def query_entity(name_entity):
    
    res_outgoing = test_graph.run("match (a{name:'%s'})-[rel]->(b) return a.name, b.name, rel" % name_entity).data()
    res_incoming = test_graph.run("match (a)-[rel]->(b{name:'%s'}) return a.name, b.name, rel" % name_entity).data()
    rel_outgoing, rel_incoming = [], []
    for item in res_outgoing:
        rel_outgoing.append([item['a.name'], item['rel']['type'], item['b.name']])
    for item in res_incoming:
        rel_incoming.append([item['a.name'], item['rel']['type'], item['b.name']])
    
    return rel_outgoing, rel_incoming


if __name__ == '__main__':
    # rel_triple_list = [[['汉族', '李彦宏', '民族', 'Text', '人物'], ['175cm', '李彦宏', '身高', 'Number', '人物'], ['1968年11月', '李彦宏', '出生日期', 'Date', '人物'], ['山西阳泉', '李彦宏', '出生地', '地点', '人物']], [['李彦宏', '百度', '创始人', '人物', '企业'], ['李彦宏', '百度', '董事长', '人物', '企业']], [['1999年11月', '百度', '成立日期', 'Date', '企业']], [['马东敏', '李彦宏', '妻子', '人物', '人物'], ['李彦宏', '马东敏', '丈夫', '人物', '人物']], [], [['李富贵', '李彦宏', '父亲', '人物', '人物']]]
    # rel_triple_list = [[['北宋', '苏轼', '朝代', 'Text', '历史人物']], [['贾乃亮', '李小璐', '丈夫', '人物', '人物'], ['李小璐', '贾乃亮', '妻子', '人物', '人物']], [['大飞', '直线', '作词', '人物', '歌曲'], ['深白色', '直线', '作曲', '人物', '歌曲']]]
    rel_triple_list = [[['苏辙', '苏轼', '儿子', '历史人物', '历史人物']]]
    build_graph(rel_triple_list)
    tmp = test_graph.run('match (a)-[r]->(b) return a.name, r, b.name').data()
    print("*"*50)
    print(tmp)
    print("*" * 50)
    for item in tmp:
        print(item['a.name'], item['r']['type'], item['b.name'])
    print("#"*50)
    name = input("请输入你要查询的实体的名称：")
    rel_outgoing, rel_incoming = query_entity(name)
    print("出度")
    for item in rel_outgoing:
        print(item)
    print("入度")
    for item in rel_incoming:
        print(item)
    # 后续要做的功能： 1，自定义查询； 2，将知识图谱转化为excel文件等惊醒保存
