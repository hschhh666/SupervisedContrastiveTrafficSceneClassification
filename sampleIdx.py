import numpy as np
import torch
import random
from torch.utils.data.sampler import BatchSampler, RandomSampler

class RandomBatchSamplerWithPosAndNeg(BatchSampler):
    def __init__(self, dataset, args, drop_last=False):
        self.dataset = dataset
        self.batch_size = args.batch_size
        self.nce_k = args.nce_k
        self.drop_last = drop_last
        self.sampler = RandomSampler(dataset)
        self.sample_dict = args.sample_dict
        self.random_neg_percent = args.global_neg_percent
        self.get_pos_neg_idx = getPosNegIdx(len(dataset), self.sample_dict,self.nce_k, self.random_neg_percent)
        super().__init__(self.sampler, self.batch_size, self.drop_last)
        
    def __iter__(self):
        batch = []
        anchor_idx = []
        pos_and_neg_idx = []
        for i in self.sampler:
            anchor_idx.append(i)
            pos_and_neg_idx = self.get_pos_neg_idx.get(i)
            if len(anchor_idx) == self.batch_size:
                batch = anchor_idx + pos_and_neg_idx #输出 batchSize + batchSize*(1+N)张照片，batchSize是anchor image，batchSize*(1+N)中，对于每个batch，第一个是pos，剩下N个是neg
                yield batch
                batch = []
                anchor_idx = []
                pos_and_neg_idx = []
        if len(anchor_idx) > 0 and not self.drop_last:
            batch = anchor_idx + pos_and_neg_idx
            yield batch       


class getPosNegIdx(): # 输入一个样本的索引，输出一维列表，列表中的0号元素是该样本的正样本，其它元素是该样本的负样本
    def __init__(self, n_data, sample_dict, nce_k, random_neg_percent) -> None:
        self.sample_dict = np.load(sample_dict, allow_pickle = True).item()
        self.dataIdx_set = set(range(n_data))
        self.nce_k = nce_k
        self.random_neg_percent = random_neg_percent

    def get(self,i) -> list: # 返回一个list
        pos_and_neg_idx = []
        posIdx = random.sample(self.sample_dict[i]['pos'] + [i],1) # 正样本从指定的正样本和自身augmentation中采样
        pos_and_neg_idx += posIdx

        # 负样本的构成是这样的：对比学习的通用范式是，对于某样本S而言，数据集中的所有其他样本都是它的负样本，而S自身的augmentation是它的正样本。在本实验中，我们已经手动指定了部分样本的正负样本，但为了一定程度上遵照对比学习正负样本的通用范式（这样也会有好的效果），决定对于样本S而言，其负样本有一部分来自手动指定的负样本，而另一部分则在未知关系的样本中采样，要保证至少n%的负样本是在未知关系的样本中采样来的，剩下的负样本则在指定的负样本中采样。
        
        dict_neg_num = len(self.sample_dict[i]['neg']) # 为当前样本标注了这么多负样本
        unknow_relation_neg_num = int(self.random_neg_percent * self.nce_k) # 需要从未知关系的样本中至少采样这么多负样本
        unknow_relation_list = list(self.dataIdx_set - set(self.sample_dict[i]['neg']) - set(self.sample_dict[i]['pos']) - set([i])) # 未知关系的样本列表

        if dict_neg_num + unknow_relation_neg_num > self.nce_k: # 当前指定的负样本个数已经足够了
            dict_neg_num = self.nce_k - unknow_relation_neg_num
        else:                                           # 当前指定的负样本不够时
            unknow_relation_neg_num = self.nce_k - dict_neg_num
        
        if unknow_relation_neg_num > len(unknow_relation_list): # 还有一种可能，就是未知关系的样本数量不够了，这说明标注已经非常充分了
            unknow_relation_neg_num = len(unknow_relation_list)
            dict_neg_num = self.nce_k - unknow_relation_neg_num

        negIdx = random.sample(self.sample_dict[i]['neg'], dict_neg_num) # 从指定的负样本中采样
        negIdx += random.sample(unknow_relation_list, unknow_relation_neg_num) # 在全局采样

        pos_and_neg_idx += negIdx
        return pos_and_neg_idx
