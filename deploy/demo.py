# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/27 10:12
# @File    : demo.py

"""
file description:：

"""
import torch
import sys
sys.path.append('/home/xieminghui/Projects/MultiHeadJointEntityRelationExtraction_simple/')  # 添加路径

from data_loader.data_process import ModelDataPreparation
from mains.trainer import Trainer
from utils.config import Config, USE_CUDA
from modules.joint_model import JointModel
import json


def test():
    path_test = './test.json'
    PATH_MODEL = '../models/27m-p0.83f0.83n2.32r0.66.pth'
    config = Config()
    num_sample_test = 0
    with open(path_test, 'r', encoding='utf-8') as f:
        for line in f:
            num_sample_test += 1
    config.batch_size = num_sample_test
    model = JointModel(config)
    model_dict = torch.load(PATH_MODEL)
    model.load_state_dict(model_dict['state_dict'])
    
    data_processor = ModelDataPreparation(config)
    _, _, test_loader = data_processor.get_train_dev_data(path_test=path_test)
    trainer = Trainer(model, config, test_dataset=test_loader)
    texts, token_pred, rel_triple_list = trainer.predict()
    for i in range(len(texts)):
        print(texts[i])
        print("token_pred: {}".format(token_pred[i]))
        print("提取的关系三元组:\n {}".format(rel_triple_list[i]))
        print("*"*50)
    print("所有关系三元组:\n {}".format(rel_triple_list))
    

if __name__ == '__main__':
    test()
    
    
    
    
    
    
    
    
    


