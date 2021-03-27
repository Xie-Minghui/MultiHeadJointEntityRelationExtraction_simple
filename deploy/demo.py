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

def test():
    path_test = './test.json'
    PATH_MODEL = '../models/27m-p0.83f0.83n2.32r0.66.pth'
    config = Config()
    config.batch_size = 1
    model = JointModel(config)
    model_dict = torch.load(PATH_MODEL)
    model.load_state_dict(model_dict['state_dict'])
    
    data_processor = ModelDataPreparation(config)
    _, _, test_loader = data_processor.get_train_dev_data(path_test=path_test)
    trainer = Trainer(model, config, test_dataset=test_loader)
    rel_triple = trainer.predict()
    print("提取得到的关系三元组:\n {}".format(rel_triple))
    

if __name__ == '__main__':
    test()
    
    
    
    
    
    
    
    
    


