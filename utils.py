import os
from random import random

import torch as torch
import tqdm as tqdm
from torch.nn.utils import rnn

from torch.utils.data import Dataset, BatchSampler, RandomSampler


def seed_everything(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def make_data_from_txt(path, max_data_size, tokenizer):
    data = list()
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(0, min(len(lines), max_data_size)):
        s1, s2 = lines[i].replace('\n', '').split('\t')
        data.append(tuple([tokenizer.encode(s1), tokenizer.encode(s2)]))
    return data

def make_error_data_from_txt(path, max_data_size, tokenizer):
    from m2utils.convert_from_m2 import Sentence
    list = Sentence.readfile(path)
    data = []
    for i, s in enumerate(list):
        str_array = s.question.split(' ')
        # adding [CLS] flag  and one more for insertion after period
        error_pos = [0] * (len(str_array) + 2)
        for error in s.corrections:
            # adding [CLS] flag
            error_pos[error[0] + 1] = 1

        data.append(tuple([tokenizer.encode(s.question), tokenizer.encode(s.getAnswer()), error_pos]))
    return data




class GECDataset(Dataset):

    def __init__(self, train_data):
        self.data = train_data

    def __getitem__(self, idx):
        src, tgt, err = self.data[idx]
        src = torch.LongTensor(src)
        tgt = torch.LongTensor(tgt)
        err =  torch.LongTensor(err)
        return src, tgt, err

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
        err_list = list()

        # sampler is RandomSampler
        for i in self.sampler:
            self.count += 1
            src, tgt, err = self.sampler.data_source[i]
            src_list.append(src)
            tgt_list.append(tgt)
            err_list.append(err)
            if self.count % self.batch_size == 0:
                assert len(src_list) == len(tgt_list)

                # fill with padding for max sentence length of src_list, tgt_list
                consolidated_list = src_list + tgt_list + err_list
                size = len(src_list)
                padded_consolidated_list = rnn.pad_sequence(consolidated_list, batch_first=True, padding_value=self.pad_id)
                src = padded_consolidated_list[:size]
                tgt = padded_consolidated_list[size:size * 2]
                err = padded_consolidated_list[size*2:]

                src_list.clear()
                tgt_list.clear()
                err_list.clear()
                yield src, tgt, err

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
