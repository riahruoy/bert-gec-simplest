import os
from random import random

import torch as torch
import tqdm as tqdm
from torch.nn.utils import rnn

from torch.utils.data import Dataset, BatchSampler, RandomSampler
import nltk

def seed_everything(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

pos_padid = 0
pos_stoi = {'[PAD]': 0}

def pos_encode(text):
    global pos_stoi
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    tag_list = []
    for v in tagged:
        tag = v[1]
        if tag not in pos_stoi:
            pos_stoi[tag] = len(pos_stoi)
        tag_list += [pos_stoi[tag]]
    return tag_list


def make_data_from_txt(path, max_data_size, tokenizer):
    data = list()
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(0, min(len(lines), max_data_size)):
        s1, s2 = lines[i].replace('\n', '').split('\t')
        pos = pos_encode(s1)
        data.append(tuple([tokenizer.encode(s1), tokenizer.encode(s2), pos]))
    return data


class GECDataset(Dataset):

    def __init__(self, train_data):
        self.data = train_data

    def __getitem__(self, idx):
        src, tgt, pos = self.data[idx]
        src = torch.LongTensor(src)
        tgt = torch.LongTensor(tgt)
        pos = torch.LongTensor(pos)
        return src, tgt, pos

    def __len__(self):
        return len(self.data)


class BalancedDataLoader(BatchSampler):

    def __init__(self, data: Dataset, pad_id: int, batch_size):
        super().__init__(RandomSampler(data), batch_size, True)
        self.pad_id = pad_id
        self.count = 0

    def __iter__(self):
        src_list = list()
        tgt_list = list()
        pos_list = list()
        # sampler is RandomSampler
        for i in self.sampler:
            self.count += 1
            src, tgt, pos = self.sampler.data_source[i]
            src_list.append(src)
            tgt_list.append(tgt)
            pos_list.append(pos)
            if self.count % self.batch_size == 0:
                assert len(src_list) == len(tgt_list)

                # fill with padding for max sentence length of src_list, tgt_list
                src_tgt_list = src_list + tgt_list + pos_list
                size = len(src_list)
                padded_src_tgt_list = rnn.pad_sequence(src_tgt_list, batch_first=True, padding_value=self.pad_id)
                src = padded_src_tgt_list[:size]
                tgt = padded_src_tgt_list[size:size * 2]
                pos = padded_src_tgt_list[size * 2 : ]

                src_list.clear()
                tgt_list.clear()
                pos_list.clear()
                yield src, tgt, pos

def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (1, size, size)
    import numpy as np
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    def __init__(self, source: torch.Tensor, target: torch.Tensor = None, pad: int = 0):

        self.source = source
        self.source_mask = (source != pad)

        if target is not None:
            self.target = target
            self.target_y = target
            self.target_mask = self.make_std_mask(self.target, pad)
            self.n_tokens = (self.target != pad).sum()

    @staticmethod
    def make_std_mask(target: torch.Tensor, pad: int) -> torch.Tensor:
        mask = (target != pad).unsqueeze(-2)
        mask = mask & subsequent_mask(target.size(-1)).type_as(mask)
        return mask
