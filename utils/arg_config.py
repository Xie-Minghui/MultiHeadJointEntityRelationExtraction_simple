# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/4/14 20:32
# @File    : arg_config.py

"""
file description:：

"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="mutable parameters")
    parser.add_argument('-encode', '--encode_name', help='choose from gru, bert, albert, joint_model.py', default='gru')
    # bool变量，由于parser的原因，使用int类型，非0的表示True
    parser.add_argument('-pre', '--use_pred_embedding', help='whether to use pre-trained embeddings', type=int, default=0)
    # parser.add_argument('-resume', '--use_resume', help='whether to resume training', type=int, default=0)
    parser.add_argument('-adv', '--use_adv', help='whether to use adc training, joint_model_adv.py', type=int, default=0)

    args = vars(parser.parse_args())
    
    return args


args = parse_args()