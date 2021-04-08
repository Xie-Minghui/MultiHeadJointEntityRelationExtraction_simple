# MultiHeadJointEntityRelationExtraction

PyTorch code for MultiHead:"Joint entity recognition and relation extraction as a multi-head selection problem".I use Chinese dataset to achieve chinese entity relation extraction.

For a description of the model and experiment, see paper https://arxiv.org/abs/1804.07847.

![image-20210408161005676](C:\Users\xmh\AppData\Roaming\Typora\typora-user-images\image-20210408161005676.png)



### Requirements

- torch==1.4.0+cu100
- cuda=10.0
- cudnn=7603
- pytorch-crf==0.7.2
- transformers==4.3.3
- tqdm==4.59.0
- seqeval==0.0.10
- tensorboard

### Dataset

Baidu CCKS2019 Competition

Download link: https://ai.baidu.com/broad/download?dataset=dureader

### Train

```
cd Projects/MultiHeadJointEntityRelationExtraction_simple/mains
python3 trainer.py
```

### Deployment

```
cd Projects/MultiHeadJointEntityRelationExtraction_simple/deploy_flask
python3 manage.py
```

Results:

![image-20210408162239231](C:\Users\xmh\AppData\Roaming\Typora\typora-user-images\image-20210408162239231.png)