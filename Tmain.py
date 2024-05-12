# -*- coding: UTF-8 -*-

MODELNAME = './mongolian'

import random
from tqdm import tqdm
from MonBigTool import IS_levenshtein_distance_and_operations
from MonBigTool import levenshtein_distance_and_operations
from MonBigTool import MonBigTool,MASKmodel

mASKmodel = MASKmodel(MODELNAME)
monBigTool = MonBigTool()
WordsDict = monBigTool.getWordsDict()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODELNAME, use_fast=False)

texts = []
labels = []

MASK =  monBigTool.getMASK()
for i in tqdm(MASK):

    sen = []
    _labels = []
    token_labels = []
    for j in i['sen']:
        if j == '<>':
            i['word'].pop(0)
            sen.append( i['word'].pop(0))
            _labels.append(1)
        else:
            sen.append(j)
            _labels.append(0)
    
    sentence = ' '.join(sen)

    # sentence_length = tokenizer(sentence)['input_ids']
    # tokens = tokenizer.convert_ids_to_tokens(sentence_length)
    # tokens = [token.replace('▁', '') for token in tokens]
    # tokens = [token.replace('#', '') for token in tokens]

    # tokens2 = ['[CLS]']
    token_labels = [0]
    for j in range(len(sen)):
        aAaaa = tokenizer.tokenize(sen[j])
        token_labels += ([_labels[j]] * len(aAaaa))
        # aAaaa = [token.replace('▁', '') for token in aAaaa]
        # aAaaa = [token.replace('#', '') for token in aAaaa]
        # tokens2 += aAaaa
    # tokens2.append('[SEP]')
    token_labels.append(0)

    if len(token_labels) > 32:
        continue
    texts.append(sentence)
    labels.append(token_labels)

    
    # print(sen)
    # print(labels)

    # 判断 token 和 token2 是否一致
    # if len(tokens2) != len(tokenlabels):
    #     print('error')
    #     print(' '.join(sen))
    #     print(tokens)
    #     print(tokens2)
    #     print('---------------------')
    #     continue


from torch.utils.data import Dataset
import torch

from transformers import BertTokenizer
    # 读取

class SpellingErrorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        # 对文本进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
        )
        # 替换 32000 为 0
        encoding['input_ids'][encoding['input_ids'] == 32000] = 0
        # print(encoding['input_ids'].flatten().max())
        # print(encoding['input_ids'].flatten().min())
        # print(sentence)
        # print(token_labels)
        # print('---------------------')
        # 将标签对齐到编码的长度
        labels = labels + [0] * (self.max_len - len(labels))

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
tokenizer = AutoTokenizer.from_pretrained(MODELNAME, use_fast=False)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# 创建分词器


dataset = SpellingErrorDataset(texts, labels, tokenizer, max_len=32)


train_dataset = dataset
test_dataset = dataset

from transformers import BertForTokenClassification, Trainer, TrainingArguments

# 加载预训练模型，指定要进行分类的类别数量
model = BertForTokenClassification.from_pretrained(MODELNAME, num_labels=2)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 总的训练轮数
    per_device_train_batch_size=16,  # 每个设备的训练批次大小
    per_device_eval_batch_size=64,   # 每个设备的评估批次大小
    warmup_steps=500,                # 预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
)

# 定义训练器
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset            # 评估数据集
)

# 开始训练
trainer.train()