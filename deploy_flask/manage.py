# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/4/2 10:40
# @File    : manage.py

"""
file description:：

"""

from flask import Flask, render_template, request, jsonify
from mains.trainer import Trainer
import json
from deploy.demo import test


def sent_split(text):
    sentences_list = text.split('\n')
    sentences_list = [x for x in sentences_list if x.strip()!='']
    
    return sentences_list


def flask_server():
    
    app = Flask(__name__)
    
    @app.route("/")
    def index():
        return render_template("index.html", version='V 0.1.2')

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