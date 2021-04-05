# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/4/2 10:40
# @File    : manage.py

"""
file description:：

"""

from flask import Flask, render_template, request, jsonify
import json
from deploy.demo import test
from py2neo import Graph

# set up authentication parameters
# authenticate("localhos t:7474", "neo4j", "root")
# connect to authenticated graph database
# graph = Graph("http://localhost:7474/db/data/")
graph = Graph("http://localhost:7474",username="neo4j", password="root")

def build_nodes(node_record):
    data = {"id": str(node_record.n._id), "label": next(iter(node_record.n.labels))}
    data.update(node_record.n.properties)

    return {"data": data}


def build_edges(relation_record):
    data = {"source": str(relation_record.r.start_node._id),
            "target": str(relation_record.r.end_node._id),
            "relationship": relation_record.r.rel.type}

    return {"data": data}


def sent_split(text):
    sentences_list = text.split('\n')
    sentences_list = [x for x in sentences_list if x.strip()!='']
    
    return sentences_list


def flask_server():
    
    app = Flask(__name__)
    
    @app.route("/")
    def index():
        return render_template("index_neo4j.html", version='V 0.1.2')

    @app.route("/query", methods=["POST"])
    def query():
        res = {}
        text = request.values['text']
        if not text:
            res["result"] = "error"
            return jsonify(res)
        setences_list = sent_split(text)
        sentences_map_list = [{"text": text} for text in setences_list]
        path_test = '../deploy/test.json'
        with open(path_test, 'w', encoding='utf-8') as f:  # 清空文件内容
            pass
        for sentence in sentences_map_list:
            with open(path_test, 'a+', encoding='utf-8') as f:
                json.dump(sentence, f, ensure_ascii=False)
            with open(path_test, 'a+', encoding='utf-8') as f:
                f.write('\n')
        rel_triple_list = test()
        res['result'] = rel_triple_list
        
        return jsonify(res)
    
    app.run(debug=True)
    

if __name__ == '__main__':
    flask_server()