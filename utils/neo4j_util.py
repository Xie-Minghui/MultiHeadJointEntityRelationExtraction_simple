# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/4/19 17:40
# @File    : neo4j_util.py

"""
file description:：

"""
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher

def build_graph(rel_triple_list, neo4j_graph):
    matcher = NodeMatcher(neo4j_graph)
    for rel_triple in rel_triple_list:
        if rel_triple is not None:
            for item in rel_triple:
                node_s = list(matcher.match(item[0], name=item[0]))
                node_o = list(matcher.match(item[1], name=item[1]))
                
                if len(node_s) <= 0:
                    subject = Node(item[0], label=item[3], name=item[0])
                    neo4j_graph.create(subject)
                else:
                    subject = node_s[0]
                if len(node_o) <= 0:
                    object = Node(item[1], label=item[4], name=item[1])
                    neo4j_graph.create(object)
                else:
                    object = node_o[0]
                
                rel = Relationship(subject, item[2], object)
                rel['type'] = item[2]
                neo4j_graph.create(rel)


# 创建节点
def entity_query(name_entity, neo4j_graph):
    name_entity = ''.join(name_entity.split())  # 去除制表符
    res_outgoing = neo4j_graph.run("match (a{name:'%s'})-[rel]->(b) return a.name, b.name, rel" % name_entity).data()
    res_incoming = neo4j_graph.run("match (a)-[rel]->(b{name:'%s'}) return a.name, b.name, rel" % name_entity).data()
    rel_outgoing, rel_incoming = [], []
    for item in res_outgoing:
        rel_outgoing.append([item['a.name'], item['b.name'], item['rel']['type']])
    for item in res_incoming:
        rel_incoming.append([item['a.name'], item['b.name'], item['rel']['type']])
    
    return rel_outgoing, rel_incoming

# 创建节点
def rel_query(name_rel, neo4j_graph):
    name_rel = ''.join(name_rel.split())  # 去除制表符
    relations = neo4j_graph.run("match (a)-[rel{type:'%s'}]->(b) return a.name, b.name, rel" % name_rel).data()
    # res_incoming = neo4j_graph.run("match (a)-[rel]->(b{name:'%s'}) return a.name, b.name, rel" % name_entity).data()
    # rel_outgoing, rel_incoming = [], []
    # for item in res_outgoing:
    #     rel_outgoing.append([item['a.name'], item['b.name'], item['rel']['type']])
    # for item in res_incoming:
    #     rel_incoming.append([item['a.name'], item['b.name'], item['rel']['type']])
    # matcher_rel = RelationshipMatcher(neo4j_graph)
    # relation = list(matcher_rel.match(type=name_rel))
    rel_list = []
    for item in relations:
        rel_list.append([item['a.name'], item['b.name'], item['rel']['type']])
    rel_list = [rel_list]
    print(rel_list)
    return rel_list
