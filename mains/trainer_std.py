# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/4/13 22:28
# @File    : trainer_std.py

"""
file description:：

"""
import sys

sys.path.append('/home/xieminghui/Projects/MultiHeadJointEntityRelationExtraction_simple/')  # 添加路径

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.config import Config, USE_CUDA
from modules.joint_model import JointModel
from data_loader.data_process import ModelDataPreparation
import math

from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
import numpy as np
import codecs
import random
# import neptune
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../record/log')  # tensorboard日志文件的存储目录


class Trainer:
    def __init__(self,
                 model,
                 config,
                 train_dataset=None,
                 dev_dataset=None,
                 test_dataset=None,
                 token2id=None
                 ):
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.token2id = token2id
        
        # 初始优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        # 学习率调控
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5,
                                                                    patience=8, min_lr=1e-5, verbose=True)
        if USE_CUDA:
            self.model = self.model.cuda()
        
        self.get_id2rel()

        self.num_sample_total = len(train_loader) * self.config.batch_size
    
    def get_id2rel(self):
        self.id2rel = {}
        for i, rel in enumerate(self.config.relations):
            self.id2rel[i] = rel
        self.id2token_type = {}
        for i, token_type in enumerate(self.config.token_types):
            self.id2token_type[i] = token_type
        
    def controller(self):
        self.print_model()
        precision_score_final_eval_best = 0
        for epoch in range(self.config.epochs):
            print("Epoch: {}".format(epoch))
            self.train(epoch)
            
            if (epoch + 1) % 1 == 0:
                self.predict_sample()

            if (epoch + 1) % 1 == 0:
                ner_loss_final_eval, rel_loss_final_eval, f1_ner_final_eval, precision_score_final_eval = self.evaluate(epoch)
                # 模型保存模块
                if epoch > 16 and precision_score_final_eval > precision_score_final_eval_best:
                    precision_score_final_eval_best = precision_score_final_eval
                    torch.save({
                        'epoch': epoch + 1, 'state_dict': model.state_dict(), 'precision_best': precision_score_final_eval_best,
                        'f1_best': f1_ner_final_eval,
                        'optimizer': self.optimizer.state_dict(),
                    },
                        self.config.checkpoint_path + str(epoch) + 'm-' + 'p' + str(
                            "%.2f" % precision_score_final_eval) +
                        'f' + str("%.2f" % f1_ner_final_eval) + 'n' + str(
                            "%.2f" % ner_loss_final_eval) +
                        'r' + str("%.2f" % rel_loss_final_eval) + '.pth'
                    )
    
    def recorder(self, epoch, ner_loss_final, rel_loss_final, f1_ner_final, precision_score_final, mode):
        # 实验效果记录模块 tensorboard 记录代码
        writer.add_scalar('Loss/'+mode+'_ner_loss', ner_loss_final, epoch)
        writer.add_scalar('Loss/'+mode+'_rel_loss', rel_loss_final, epoch)
        writer.add_scalar('Accuracy/'+mode+'_ner_f1', f1_ner_final, epoch)
        writer.add_scalar('Accuracy/'+mode+'_rel_ps', precision_score_final, epoch)

        # pbar.set_description('TRAIN LOSS: {}'.format(loss_total/self.num_sample_total))

        # neptune 记录代码
        # neptune.log_metric("train ner loss", loss_ner_total/self.num_sample_total)
        # neptune.log_metric("train ner f1 score", f1_ner_total/self.num_sample_total*self.config.batch_size)
        # neptune.log_metric("train rel precission score", correct_score_total / len(self.train_dataset))
    
    def update(self, loss):
        self.optimizer.zero_grad()
        # loss_ner.backward(retain_graph=True)
        # loss_rel.backward()
        loss.backward()  # retain_graph=True
        self.optimizer.step()
    
    def print_model(self):
        # for name, parameters in self.model.named_parameters():
        #     print(name, " : ", parameters.size())
        print("Total")
        self.get_parameters_number()
    
    def get_parameters_number(self):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            "Total number of parameters is {0}M, Trainable number is {1}M".format(total_num / 1e6, trainable_num / 1e6))
    
    def train(self, epoch):
        print('STARTING TRAIN...')
        pbar = tqdm(enumerate(self.train_dataset), total=len(self.train_dataset))
        loss_total, loss_ner_total, loss_rel_total, f1_ner_total, correct_score_total = 0, 0, 0, 0, 0
        for i, data_item in pbar:
            loss_ner, loss_rel, pred_ner, pred_rel = self.model(data_item)
            # loss_ner, loss_rel, pred_ner, pred_rel, f1_ner = self.train_batch(data_item)
            pred_token_type = self.restore_ner(pred_ner, data_item['mask_tokens'])
            f1_ner = f1_score(data_item['token_type_origin'], pred_token_type)
            
            loss_total += (float(loss_ner) + float(loss_rel))
            loss_ner_total += float(loss_ner)
            loss_rel_total += float(loss_rel)
            f1_ner_total += f1_ner
            
            pred_rel_max = -torch.argmax(pred_rel, dim=-1)
            one = torch.ones_like(pred_rel_max)
            pred_rel_max = torch.where(pred_rel_max > -0.5, one, pred_rel_max)
            pred_rel_max = -pred_rel_max
            correct = torch.eq(pred_rel_max, data_item['pred_rel_matrix'].data).cpu().sum().numpy()
            # correct = torch.sum(correct)
            num_non_zero = data_item['pred_rel_matrix'].nonzero().size(0)
            correct_score = correct / num_non_zero
            correct_score_total += correct_score

            loss = (loss_ner + loss_rel)
            self.update(loss)
        
        ner_loss_final_train = loss_ner_total / self.num_sample_total
        rel_loss_final_train = loss_rel_total / self.num_sample_total
        f1_ner_final_train = f1_ner_total / self.num_sample_total * self.config.batch_size
        precision_score_final_train = correct_score_total / len(self.train_dataset)
        print(
            "train ner loss: {0}, rel loss: {1}, f1 score: {2}, precission score: {3}".format(ner_loss_final_train,
                                                                                              rel_loss_final_train,
                                                                                              f1_ner_final_train,
                                                                                              precision_score_final_train))

        self.recorder(epoch, ner_loss_final_train, rel_loss_final_train, f1_ner_final_train, precision_score_final_train,
                     mode='train')
    
    def evaluate(self, epoch):
        print('STARTING EVALUATION...')
        self.model.train(False)
        pbar_dev = tqdm(enumerate(self.dev_dataset), total=len(self.dev_dataset))
        
        loss_total, loss_ner_total, loss_rel_total, f1_ner_total, correct_score_total = 0, 0, 0, 0, 0
        samle_num = len(self.dev_dataset) * self.config.batch_size
        for i, data_item in pbar_dev:
            loss_ner, loss_rel, pred_ner, pred_rel = self.model(data_item, is_eval=True)
            pred_token_type = self.restore_ner(pred_ner, data_item['mask_tokens'])
            f1_ner = f1_score(data_item['token_type_origin'], pred_token_type)
            f1_ner_total += f1_ner
            # loss_ner_total += loss_ner
            # loss_rel_total += loss_rel
            loss_ner_total += float(loss_ner)  # 使用float类型转换去掉梯度，可以节省几百兆的显存
            loss_rel_total += float(loss_rel)
            
            pred_rel_max = -torch.argmax(pred_rel, dim=-1)
            one = torch.ones_like(pred_rel_max)
            pred_rel_max = torch.where(pred_rel_max > -0.5, one, pred_rel_max)
            pred_rel_max = -pred_rel_max
            correct = torch.eq(pred_rel_max, data_item['pred_rel_matrix'].data).cpu().sum().numpy()
            # correct = torch.sum(correct)
            num_non_zero = data_item['pred_rel_matrix'].nonzero().size(0)
            correct_score = correct / num_non_zero
            correct_score_total += correct_score
        
        ner_loss_final_eval = loss_ner_total / samle_num
        rel_loss_final_eval = loss_rel_total / samle_num
        f1_ner_final_eval = f1_ner_total / len(self.dev_dataset)
        precision_score_final_eval = correct_score_total / len(self.dev_dataset)
        print("eval ner loss: {0}, rel loss: {1}, f1 score: {2}, precission score: {3}".format(ner_loss_final_eval,
                                                                                               rel_loss_final_eval,
                                                                                               f1_ner_final_eval,
                                                                                               precision_score_final_eval))
        self.model.train(True)
        self.recorder(epoch, ner_loss_final_eval, rel_loss_final_eval, f1_ner_final_eval, precision_score_final_eval,
                      mode='eval')
        
        return ner_loss_final_eval, rel_loss_final_eval, f1_ner_final_eval, precision_score_final_eval
    
    def predict(self):
        '''
        和下面predict_sample的区别：predict接收的只有一个批次，用于实际部署时候的预测。
        predict_sample接受的是predict.json的所有样本，然后从中随机选择一个样本进行输出，用于训练的时候查看模型效果。
        :return:
        :rtype:
        '''
        print('STARTING TESTING...')
        self.model.train(False)
        pbar = tqdm(enumerate(self.test_dataset), total=len(self.test_dataset))
        for i, data_item in pbar:
            pred_ner, pred_rel = self.model(data_item, is_test=True)
        # pred_ner, pred_rel = pred_ner[0], pred_rel[0]
        pred_rel_list = [[] for _ in range(self.config.batch_size)]
        loc = pred_rel.nonzero()  # 得到关系矩阵中1的位置
        for item in loc:
            item = item.cpu().numpy()
            if math.fabs(item[3]) < 0.1:  # 排除空关系
                continue
            pred_rel_list[item[0]].append([item[1], item[2], self.id2rel[item[3]]])
        texts = [text for text in data_item['text']]
        token_pred = self.restore_ner(pred_ner, data_item['mask_tokens'])
        self.model.train(True)
        rel_triple_list = []
        for i in range(self.config.batch_size):
            rel_triple = self.convert2StandardOutput(data_item, i, token_pred[i], pred_rel_list[i])
            rel_triple_list.append(rel_triple)
        # print("提取得到的关系三元组:\n {}".format(rel_triple))
        return texts, token_pred, rel_triple_list
    
    def predict_sample(self):
        print('STARTING TESTING...')
        self.model.train(False)
        pbar = tqdm(enumerate(self.test_dataset), total=len(self.test_dataset))
        data_item0 = None
        for i, data_item in pbar:
            # pred_ner, pred_rel, atten_weights = self.model(data_item, is_test=True)
            pred_ner, pred_rel = self.model(data_item, is_test=True)
            if random.random() > 0.7:
                data_item0 = data_item
                pred_ner0, pred_rel0 = pred_ner, pred_rel
        
        if data_item0 is None:
            data_item0 = data_item
            pred_ner0, pred_rel0 = pred_ner, pred_rel
        x = random.randint(0, 15)
        pred_ner, pred_rel = pred_ner0[x], pred_rel0[x]
        pred_rel_list = []
        length = len([c for c in data_item0['text'][x]])
        
        loc = pred_rel.nonzero()
        for item in loc:
            item = item.cpu().numpy()
            if math.fabs(item[2]) < 0.1:
                continue
            if item[0] >= length or item[1] >= length:  # 预测不能超过句子长度
                continue
            pred_rel_list.append([item[0], item[1], self.id2rel[item[2]]])
        token_pred = []
        cnt = 0
        for i in pred_ner:
            if cnt >= length:
                break
            token_pred.append(self.id2token_type[i])
            cnt += 1
        # token_pred = self.restore_ner(pred_ner, data_item0['mask_tokens'])  # restore_ner只针对批次预测结果
        print("token_pred: {}".format(token_pred))
        # print("attention_weights:{}".format(atten_weights))
        print(data_item0['text'][x])
        # print(data_item0['spo_list'][x])
        print("pred_rel_list: {}".format(pred_rel_list))
        self.model.train(True)
        rel_triple = self.convert2StandardOutput(data_item0, x, token_pred, pred_rel_list)
        print("提取得到的关系三元组:\n {}".format(rel_triple))
    
    def restore_ner(self, pred_ner, mask_tokens):  # 将预测的结果还原成命名实体识别的结果
        pred_token_type = []
        for i in range(len(pred_ner)):
            list_tmp = []
            for j in range(len(pred_ner[i])):
                if mask_tokens[i, j] == 0:
                    break
                list_tmp.append(self.id2token_type[pred_ner[i][j]])
            pred_token_type.append(list_tmp)
        
        return pred_token_type
    
    def convert2StandardOutput(self, data_item, loc, token_pred, pred_rel_list):
        '''
        根据文本，命名实体识别结果和关系抽取结果，得到最终的关系三元组
        '''
        subject_all, object_all, rel_all = [], [], []
        text = [c for c in data_item['text'][loc]]
        for item in pred_rel_list:
            subject, object, rel = [], [], []
            s_start, o_start = item[0], item[1]
            if s_start == o_start:  # 防止自己和自己构成关系
                continue
            if s_start >= len(text) or o_start >= len(text) or token_pred[s_start][0] != 'B' or token_pred[o_start][
                0] != 'B':
                continue
            subject.append(text[s_start])
            object.append(text[o_start])
            s_start += 1
            o_start += 1
            while s_start < len(text) and (token_pred[s_start][0] == 'I'):  # or token_pred[s_start][0] == 'B'
                subject.append(text[s_start])
                s_start += 1
            while o_start < len(text) and (token_pred[o_start][0] == 'I'):  # or token_pred[o_start][0] == 'B'
                object.append(text[o_start])
                o_start += 1
            subject = ''.join(subject)
            object = ''.join(object)
            is_repeated = False
            if rel_all is not None:
                for i in range(len(rel_all)):
                    if subject == subject_all[i] and object == object_all[i] and item[2] == rel_all[i]:
                        is_repeated = True
                        break
            if not is_repeated:
                subject_all.append(subject)
                object_all.append(object)
                rel_all.append(item[2])
        rel_triple = [[] for _ in range(len(rel_all))]
        for i in range(len(rel_all)):
            rel_triple[i] = [object_all[i], subject_all[i], rel_all[i]]
        return rel_triple


def get_embedding_pre():
    word2id = {}
    with codecs.open('../data/vec.txt', 'r', encoding='utf-8') as f:
        cnt = 0
        for line in f.readlines():
            word2id[line.split()[0]] = cnt
            cnt += 1
    
    word2vec = {}
    with codecs.open('../data/vec.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word2vec[line.split()[0]] = list(map(eval, line.split()[1:]))
        unkown_pre = []
        unkown_pre.extend([1] * 100)
    embedding_pre = []
    embedding_pre.append(unkown_pre)
    for word in word2id:
        if word in word2vec:
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unkown_pre)
    embedding_pre = np.array(embedding_pre)
    return embedding_pre


if __name__ == '__main__':
    # neptune.init(
    #     api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNTM3OTQzY2ItMzRhNC00YjYzLWJhMTktMzI0NTk4NmM4NDc3In0=',
    #     project_qualified_name='mangopudding/MultiHeadJointEntityRelationExtraction-simple')
    # neptune.create_experiment('rel_train')
    config = Config()
    if config.use_attention:
        print("use attention")
    if config.use_jieba:
        print("use jieba")
    print("do not use tanh")
    print("学习率：{}".format(config.lr))
    print("teach_rate: {}".format(config.teach_rate))
    if config.use_pred_embedding:
        embedding_pre = get_embedding_pre()
    else:
        embedding_pre = None
    data_processor = ModelDataPreparation(config)
    train_loader, dev_loader, test_loader = data_processor.get_train_dev_data(
        '../data/train_data.json',
        '../data/dev_data.json',
        '../data/predict.json')
    print(len(train_loader), len(dev_loader), len(test_loader))
    model = JointModel(config, embedding_pre)
    # train_loader, dev_loader, test_loader = data_processor.get_train_dev_data('../data/train_data_small.json')
    trainer = Trainer(model, config, train_loader, dev_loader, test_loader, data_processor.token2id)
    trainer.controller()