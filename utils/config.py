# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/3 20:42
# @File    : config.py

"""
file description:：

"""
import torch

if torch.cuda.is_available():
    USE_CUDA = True
    print("USE_CUDA....")
else:
    USE_CUDA = False


class Config:
    def __init__(self,
                 lr=1e-3,
                 epochs=100,
                 vocab_size=16116,  #16116,21128
                 embedding_dim=100,
                 hidden_dim_lstm=156,  # 未加载 64, bert 384, albert 156,
                 num_layers=3,  # 上去会超过内存
                 batch_size=16,
                 layer_size=64,
                 token_type_dim=12
                 ):
        self.lr = lr
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim_lstm = hidden_dim_lstm
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer_size = layer_size
        self.token_type_dim = token_type_dim
        self.relations = ["N", '丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地',
                '出生日期', '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑',
                '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高',
                '连载网站', '邮政编码', '面积', '首都']
        self.num_relations = len(self.relations)
        self.token_types_origin = ['Date', 'Number', 'Text', '书籍', '人物', '企业', '作品', '出版社', '历史人物', '国家', '图书作品', '地点', '城市', '学校', '学科专业',
         '影视作品', '景点', '机构', '歌曲', '气候', '生物', '电视综艺', '目', '网站', '网络小说', '行政区', '语言', '音乐专辑']
        self.token_types = self.get_token_types()
        self.num_token_type = len(self.token_types)
        self.vocab_file = '../data/vocab.txt'
        self.max_seq_length = 256
        self.num_sample = 20480
        self.num_sample_eval = 10000  # 320

        self.dropout_embedding = 0.1  # 从0.2到0.1
        self.dropout_lstm = 0.1
        self.dropout_lstm_output = 0.1
        self.dropout_head = 0.1  # 只更改这个参数 0.9到0.5
        self.dropout_ner = 0.1
        self.use_dropout = True
        self.threshold_rel = 0.9  # 从0.7到0.95
        self.teach_rate = 0.9
        self.model_best_save_path = '../models/'
        self.checkpoint_path = '../checkpoints/'
        self.rel_emb_size = 64
        self.pad_token_id = 0
        
        self.use_adv = True
        self.use_attention = False
        self.use_pred_embedding = False
        self.alpha = 1e-3
        self.gamma = 0.5
        
        self.encode_name = 'albert'
        self.use_jieba = False
        
        self.use_resume = False
        self.checkpoint_path_resume = '20m-p0.79f0.81n2.39r0.87.pth'
    
    def get_token_types(self):
        token_type_bio = []
        for token_type in self.token_types_origin:
            token_type_bio.append('B-' + token_type)
            token_type_bio.append('I-' + token_type)
        token_type_bio.append('O')
        
        return token_type_bio

