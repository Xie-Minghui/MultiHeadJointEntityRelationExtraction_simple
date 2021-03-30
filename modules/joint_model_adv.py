# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/26 20:30
# @File    : joint_model_adv.py

"""
file description:：

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import numpy as np
from utils.config import USE_CUDA
# from utils.FocalLoss import Focal_loss
import math

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JointModel(nn.Module):
    def __init__(self, config, embedding_pre=None):
        super().__init__()
        setup_seed(1)
        print("use adv training")
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim_lstm
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.layer_size = config.layer_size  # self.hidden_dim, 之前这里没有改
        self.num_token_type = config.num_token_type  # 实体类型的综述
        self.config = config
        if embedding_pre is not None:  # 测试不加载词向量的情况
            print("use pretrained embeddings")
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre), freeze=False)
        else:
            self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id)
        # self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.token_type_embedding = nn.Embedding(config.num_token_type, config.token_type_dim)
        self.rel_embedding = nn.Embedding(config.num_relations, config.rel_emb_size)
        self.gru = nn.GRU(config.embedding_dim, config.hidden_dim_lstm, num_layers=config.num_layers, batch_first=True,
                          bidirectional=True, dropout=config.dropout_lstm)
        self.is_train = True
        if USE_CUDA:
            self.weights_rel = (torch.ones(self.config.num_relations) * 50).cuda()
        else:
            self.weights_rel = torch.ones(self.config.num_relations) * 50
        self.weights_rel[0] = 1
        
        if USE_CUDA:
            self.pos_weights_rel = (torch.ones(self.config.num_relations) * 20).cuda()
        else:
            self.pos_weights_rel = torch.ones(self.config.num_relations) * 20
        self.pos_weights_rel[0] = 1
        
        self.dropout_embedding_layer = torch.nn.Dropout(config.dropout_embedding)
        # self.dropout_head_layer = torch.nn.Dropout(config.dropout_head)
        # self.dropout_ner_layer = torch.nn.Dropout(config.dropout_ner)
        # self.dropout_lstm_layer = torch.nn.Dropout(config.dropout_lstm)
        self.crf_model = CRF(self.num_token_type, batch_first=True)
        
        self.ner_layer = nn.Linear(config.hidden_dim_lstm * 2, config.num_token_type)
        
        self.selection_u = nn.Linear(self.hidden_dim * 2 + self.config.token_type_dim, config.rel_emb_size)
        self.selection_v = nn.Linear(self.hidden_dim * 2 + self.config.token_type_dim, config.rel_emb_size)
        self.selection_uv = nn.Linear(2 * config.rel_emb_size, config.rel_emb_size)
        
        # self.weights_loss = [100 for i in range(config.num_relations)]
        # self.weights_loss[0] = 1
        # self.focal_loss = Focal_loss(alpha=self.weights_loss, gamma=4, num_classes=config.num_relations)
    
    def compute_loss(self,data_item, embeddings, hidden_init, is_test=False, is_eval=False):
        output_lstm, h_n = self.gru(embeddings, hidden_init)
        # output_lstm [batch, seq_len, 2*hidden_dim]  h_n [2*num_layers, batch, hidden_dim]
        # if self.config.use_dropout:
        #     output_lstm = self.dropout_lstm_layer(output_lstm)  # 用了效果变差
        # [batch_size, seq_len, num_token_type]
        ner_score = self.ner_layer(output_lstm)
        loss_ner, loss_rel = 0, 0
        # 下面是使用CFR
        if not is_test:
            log_likelihood = self.crf_model(ner_score, data_item['token_type_list'].to(torch.int64),
                                            mask=data_item['mask_tokens'])
            loss_ner = -log_likelihood
        # [batch_size, seq_len]
        pred_ner = self.crf_model.decode(ner_score)  # , mask=data_item['mask_tokens']
    
        # --------------------------Relation
        if not is_test and torch.rand(1) > self.config.teach_rate and not is_eval:
            labels = data_item['token_type_list']
        else:
            if USE_CUDA:
                labels = torch.Tensor(pred_ner).cuda()
            else:
                labels = torch.Tensor(pred_ner)
        # [batch_size, seq_len, token_type_dim]
        label_embeddings = self.token_type_embedding(labels.to(torch.int64))
        rel_input = torch.cat((output_lstm, label_embeddings), 2)
        B, L, H = rel_input.size()
        u = torch.tanh(self.selection_u(rel_input)).unsqueeze(1).expand(B, L, L, -1)  # (B,L,L,R)
        v = torch.tanh(self.selection_v(rel_input)).unsqueeze(2).expand(B, L, L, -1)

        # u = self.selection_u(rel_input).unsqueeze(1).expand(B, L, L, -1)  # (B,L,L,R)
        # v = self.selection_v(rel_input).unsqueeze(2).expand(B, L, L, -1)
        uv = torch.tanh(self.selection_uv(torch.cat((u, v), dim=-1)))
        selection_logits = torch.einsum('bijh,rh->birj', uv, self.rel_embedding.weight)
        selection_logits = selection_logits.permute(0, 1, 3, 2)
        if not is_test:
            loss_rel = self.masked_BCEloss(data_item['mask_tokens'], selection_logits, data_item['pred_rel_matrix'],
                                           self.weights_rel)  # 要把分类放在第二维度

        return loss_ner, loss_rel, pred_ner, selection_logits
    
    def forward(self, data_item, is_test=False, is_eval=False):
        # 因为不是多跳机制，所以hidden_init不能继承之前的最终隐含态
        '''

        :param data_item: data_item = {'',}
        :type data_item: dict
        :return:
        :rtype:
        '''
        # [batch_size, seq_len, embedding_dim]
        embeddings = self.word_embedding(data_item['text_tokened'].to(torch.int64))  # 要转化为int64
        if self.config.use_dropout:
            embeddings = self.dropout_embedding_layer(embeddings)
        
        if USE_CUDA:
            hidden_init = torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim).cuda()
        else:
            hidden_init = torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim)
        if USE_CUDA:
            self.crf_model = self.crf_model.cuda()
            
        loss_ner, loss_rel, pred_ner, selection_logits = self.compute_loss(data_item, embeddings, hidden_init, is_test=is_test)
        loss_total = loss_ner + loss_rel
        if self.config.use_adv:
            raw_perturb = torch.autograd.grad(loss_total, embeddings)[0]
            normalized_per = F.normalize(raw_perturb, dim=1, p=2)
            normalized_per = F.normalize(normalized_per, dim=2, p=2)
            perturb = self.config.alpha * math.sqrt(self.config.embedding_dim) * normalized_per.detach()
            perturb_embeddings = perturb + embeddings
            loss_ner_adv, loss_rel_adv, _, _ = self.compute_loss(data_item, perturb_embeddings, hidden_init, is_test=is_test)
            loss_ner = self.config.gamma*loss_ner + (1-self.config.gamma)*loss_ner_adv
            loss_rel = self.config.gamma*loss_rel + (1-self.config.gamma)*loss_rel_adv
        
        rel_score_prob = torch.sigmoid(selection_logits)
        rel_score_prob = rel_score_prob - (self.config.threshold_rel - 0.5)  # 超过了一定阈值之后才能判断关系
        pred_rel = torch.round(rel_score_prob).to(torch.int64)
        if is_test:
            return pred_ner, pred_rel
        
        return loss_ner, loss_rel, pred_ner, pred_rel
    
    def masked_BCEloss(self, mask, selection_logits, selection_gold, weights_rel):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(3).expand(-1, -1, -1, self.config.num_relations)
        gold_predicate_matrix_one_hot = F.one_hot(selection_gold, self.config.num_relations)
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            gold_predicate_matrix_one_hot.float(),
                                                            weight=self.weights_rel,
                                                            pos_weight=self.pos_weights_rel,
                                                            reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= mask.sum()
        return selection_loss

