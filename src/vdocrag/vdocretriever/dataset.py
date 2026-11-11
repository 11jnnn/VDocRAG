from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from vdocrag.vdocretriever.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if not self.data_args.pretrain:
            self.corpus  = load_dataset(
                self.data_args.corpus_name,
                self.data_args.corpus_config,
                data_files=self.data_args.corpus_path,
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )

            self.docid2idx = {}
            if 'doc_id' in self.corpus.features:
                for idx, docid in enumerate(self.corpus['doc_id']):
                    self.docid2idx[str(docid)] = idx
            else:
                for idx in range(len(self.corpus)):
                    self.docid2idx[str(idx)] = idx

        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)


    def __getitem__(self, item) -> Tuple[str, List[str]]:
    # item 是要读取的样本编号
    # 例如 dataset[5]，就会调用 __getitem__(5)

    # 从训练数据集中取出第 item 条样本
    group = self.train_data[item]

    # 获取当前训练轮数，并转换为整数
    # 例如 epoch 为 1.6，转换后得到 1
    epoch = int(self.trainer.state.epoch)

    # 将“样本编号 + 训练种子”转换成一个哈希值
    # 后面用它决定选择哪个相关文档
    _hashed_seed = hash(item + self.trainer.args.seed)

    # 从这条训练样本中取出问题
    query = group['query']

    # 如果是预训练模式
    if self.data_args.pretrain:

        # 直接从训练样本中获取图像
        image = group['image']

    # 如果是非预训练模式
    else:

        # 获取与当前问题相关的文档 ID 列表
        relevant_docids = group['relevant_doc_ids']

        # 如果设置为“不轮换正样本文档”
        if self.data_args.positive_document_no_shuffle:

            # 始终选择列表里的第一个相关文档
            docid = relevant_docids[0]

        else:
            # 根据样本编号、种子和当前轮数，计算列表下标
            # 再选择对应的相关文档
            docid = relevant_docids[
                (_hashed_seed + epoch) % len(relevant_docids)
            ]

        # 把文档 ID 转换为语料库中的行号
        # 再取出那一行对应的图像
        image = image = self.corpus[self.docid2idx[docid]]['image']

    # 返回“问题、相关页面图像”
    return query, image
    


class EncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        if self.data_args.encode_is_query:
            self.encode_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )
        else:    
            self.encode_data = load_dataset(
                self.data_args.corpus_name,
                self.data_args.corpus_config,
                data_files=self.data_args.corpus_path,
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )

        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        
    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str]:
        data = self.encode_data[item]
        text, image = None, None
        if self.data_args.encode_is_query:
            id = data['query_id']
            text = data['query']
        else:
            id = data['doc_id']
            image = data['image']
        return id, text, image
