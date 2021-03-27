# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/3 17:28
# @File    : trainer.py

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

    def get_id2rel(self):
        self.id2rel = {}
        for i, rel in enumerate(self.config.relations):
            self.id2rel[i] = rel
        self.id2token_type = {}
        for i, token_type in enumerate(self.config.token_types):
            self.id2token_type[i] = token_type
    
    def print_model(self):
        # for name, parameters in self.model.named_parameters():
        #     print(name, " : ", parameters.size())
        
        print("Total")
        self.get_parameters_number()
    
    def get_parameters_number(self):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total number of parameters is {0}M, Trainable number is {1}M".format(total_num/1e6, trainable_num/1e6))
        
    def train(self):
        print('STARTING TRAIN...')
        self.num_sample_total = len(train_loader) * self.config.batch_size
        f1_ner_total_best = 0
        self.print_model()
        for epoch in range(self.config.epochs):
            print("Epoch: {}".format(epoch))
            pbar = tqdm(enumerate(self.train_dataset), total=len(self.train_dataset))
            loss_total, loss_ner_total, loss_rel_total, f1_ner_total, correct_score_total = 0, 0, 0, 0, 0
            for i, data_item in pbar:
                loss_ner, loss_rel, pred_ner, pred_rel, f1_ner = self.train_batch(data_item)

                loss_total += (loss_ner + loss_rel)
                loss_ner_total += loss_ner
                loss_rel_total += loss_rel
                f1_ner_total += f1_ner
                
                pred_rel_max = -torch.argmax(pred_rel, dim=-1)
                one = torch.ones_like(pred_rel_max)
                pred_rel_max = torch.where(pred_rel_max>-0.5, one, pred_rel_max)
                pred_rel_max = -pred_rel_max
                correct = torch.eq(pred_rel_max, data_item['pred_rel_matrix'].data).cpu().sum().numpy()
                # correct = torch.sum(correct)
                num_non_zero = data_item['pred_rel_matrix'].nonzero().size(0)
                correct_score = correct / num_non_zero
                correct_score_total += correct_score
                
            if (epoch+1) % 1 == 0:
                self.predict_sample()
            print("train ner loss: {0}, rel loss: {1}, f1 score: {2}, precission score: {3}".format(loss_ner_total/self.num_sample_total, loss_rel_total/self.num_sample_total,
                    f1_ner_total/self.num_sample_total*self.config.batch_size, correct_score_total / len(self.train_dataset)))
            # pbar.set_description('TRAIN LOSS: {}'.format(loss_total/self.num_sample_total))
            if (epoch+1) % 1 == 0:
                self.evaluate()
            if epoch > 8 and f1_ner_total > f1_ner_total_best:
                torch.save({
                    'epoch': epoch+1, 'state_dict': model.state_dict(), 'f1_best': f1_ner_total,
                    'optimizer': self.optimizer.state_dict(),
                },
                self.config.checkpoint_path + str(epoch) + 'm-' + 'p' + str("%.2f"%(correct_score_total / len(self.train_dataset))) +
                'f'+str("%.2f"%(f1_ner_total/self.num_sample_total*self.config.batch_size)) + 'n'+str("%.2f"%(loss_ner_total/self.num_sample_total)) +
                'r'+str("%.2f"%(loss_rel_total/self.num_sample_total)) + '.pth'
                )
    
    def train_batch(self, data_item):
        self.optimizer.zero_grad()
        loss_ner, loss_rel, pred_ner, pred_rel = self.model(data_item)
        pred_token_type = self.restore_ner(pred_ner, data_item['mask_tokens'])
        f1_ner = f1_score(data_item['token_type_origin'], pred_token_type)
        loss = (loss_ner + loss_rel)
        loss.backward()
        self.optimizer.step()
        
        return loss_ner, loss_rel, pred_ner, pred_rel, f1_ner
    
    def restore_ner(self, pred_ner, mask_tokens):
        pred_token_type = []
        for i in range(len(pred_ner)):
            list_tmp = []
            for j in range(len(pred_ner[0])):
                if mask_tokens[i, j] == 0:
                    break
                list_tmp.append(self.id2token_type[pred_ner[i][j]])
            pred_token_type.append(list_tmp)
            
        return pred_token_type
    
    def evaluate(self):
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
            loss_ner_total += loss_ner
            loss_rel_total += loss_rel

            pred_rel_max = -torch.argmax(pred_rel, dim=-1)
            one = torch.ones_like(pred_rel_max)
            pred_rel_max = torch.where(pred_rel_max > -0.5, one, pred_rel_max)
            pred_rel_max = -pred_rel_max
            correct = torch.eq(pred_rel_max, data_item['pred_rel_matrix'].data).cpu().sum().numpy()
            # correct = torch.sum(correct)
            num_non_zero = data_item['pred_rel_matrix'].nonzero().size(0)
            correct_score = correct / num_non_zero
            correct_score_total += correct_score
            # loss_total += (loss_ner + loss_rel)
        print("eval ner loss: {0}, rel loss: {1}, f1 score: {2}, precission score: {3}".format(loss_ner_total/samle_num,
            loss_rel_total/samle_num, f1_ner_total/len(self.dev_dataset), correct_score_total/ len(self.dev_dataset)))
        self.model.train(True)
        
        return loss_total / (len(self.dev_dataset) * self.config.batch_size)
    
    def predict(self):
        print('STARTING TESTING...')
        self.model.train(False)
        pbar = tqdm(enumerate(self.test_dataset), total=len(self.test_dataset))
        for i, data_item in pbar:
            pred_ner, pred_rel = self.model(data_item, is_test=True)
        # pred_ner, pred_rel = pred_ner[0], pred_rel[0]
        pred_rel_list = [[] for _ in range(self.config.batch_size)]
        loc = pred_rel.nonzero()
        for item in loc:
            item = item.cpu().numpy()
            if math.fabs(item[3]) < 0.1:  # 排除空关系
                continue
            pred_rel_list[item[0]].append([item[1], item[2], self.id2rel[item[3]]])
        texts = [text for text in data_item['text']]
        lengths = [len([c for c in data_item['text'][i]]) for i in range(self.config.batch_size)] # 测试的时候只有一个样例
        token_pred = [[] for _ in range(self.config.batch_size)]
        for i in range(len(pred_ner)):
            cnt = 0
            for id in pred_ner[i]:
                token_pred[i].append(self.id2token_type[id])
                cnt += 1
                if cnt >= lengths[i]:
                    break
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
        # for i in range(length):
        #     for j in range(length):
        #         for k in range(pred_rel.shape[2]):
        #             if math.fabs(pred_rel[i, j, k] - 1.0) < 0.1:
        #                 if k != 0:
        #                     pred_rel_list.append([i, j, self.id2rel[k]])
        loc = pred_rel.nonzero()
        for item in loc:
            item = item.cpu().numpy()
            if math.fabs(item[2]) < 0.1:
                continue
            pred_rel_list.append([item[0], item[1], self.id2rel[item[2]]])
        token_pred = []
        cnt = 0
        for i in pred_ner:
            if cnt >= length:
                break
            token_pred.append(self.id2token_type[i])
            cnt += 1
        print("token_pred: {}".format(token_pred))
        print("token_type_origin: {}".format(data_item0['token_type_origin'][0]))
        print(data_item0['text'][x])
        print(data_item0['spo_list'][x])
        print("pred_rel_list: {}".format(pred_rel_list))
        self.model.train(True)
        rel_triple = self.convert2StandardOutput(data_item0, x, token_pred, pred_rel_list)
        print("提取得到的关系三元组:\n {}".format(rel_triple))

    def convert2StandardOutput(self, data_item, loc, token_pred, pred_rel_list):
        subject_all, object_all, rel_all = [], [], []
        text = [c for c in data_item['text'][loc]]
        for item in pred_rel_list:
            subject, object, rel = [], [], []
            s_start, o_start = item[0], item[1]
            if s_start == o_start:  # 防止自己和自己构成关系
                continue
            if s_start >= len(text) or o_start >= len(text) or token_pred[s_start][0] != 'B' or token_pred[o_start][0] != 'B':
                continue
            subject.append(text[s_start])
            object.append(text[o_start])
            s_start += 1
            o_start += 1
            while s_start < len(text) and (token_pred[s_start][0] == 'I' ): # or token_pred[s_start][0] == 'B'
                subject.append(text[s_start])
                s_start += 1
            while o_start < len(text) and (token_pred[o_start][0] == 'I'): #  or token_pred[o_start][0] == 'B'
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
    config = Config()
    embedding_pre = get_embedding_pre()
    data_processor = ModelDataPreparation(config)
    train_loader, dev_loader, test_loader = data_processor.get_train_dev_data(
        '../data/train_data.json',
    '../data/dev_data.json',
    '../data/predict.json')
    print(len(train_loader),len(dev_loader), len(test_loader))
    model = JointModel(config, embedding_pre)
    # train_loader, dev_loader, test_loader = data_processor.get_train_dev_data('../data/train_data_small.json')
    trainer = Trainer(model, config, train_loader, dev_loader, test_loader, data_processor.token2id)
    trainer.train()