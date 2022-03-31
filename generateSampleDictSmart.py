import torch
from torch.utils import data
from torchvision.datasets import ImageFolder
import numpy as np
import os
import random
import argparse
from tqdm import tqdm
import time
from util import Logger
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--sample_num', type=int)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--same_range', type=int)
args = parser.parse_args()

dataset_path = args.dataset_path
sample_num = args.sample_num
same_range = args.same_range

# dataset_path = '/home/hsc/Research/TrafficSceneClassification/data/HSD_masked_selectedBy4Uniform'
# sample_num = 1750
# same_range = 3


outpath = os.path.join(dataset_path,'pos_neg_relation')
if not os.path.exists(outpath):
    os.makedirs(outpath)
log_path = os.path.join(outpath, 'sample_num%d_same_range%d.txt'%(sample_num, same_range))
out_path = os.path.join(outpath,'sample_num%d_same_range%d.npy'%(sample_num, same_range))

sys.stdout = Logger(log_path) # 把print的东西输出到txt文件中

dataset_path = os.path.join(dataset_path,'train')

torch_imageFolder = ImageFolder(dataset_path)
torch_imgs_and_targets = torch_imageFolder.imgs

sortedImgName_idx_target = []

for i, _ in enumerate(torch_imgs_and_targets):
    imgName = torch_imgs_and_targets[i][0].split('/')[-1]
    target = torch_imgs_and_targets[i][1]
    sortedImgName_idx_target.append([imgName, i, target])

sortedImgName_idx_target.sort(key= lambda x : x[0]) # list的中的每个元素都是list，依次保存着图像名（不含路径）、在torch中的索引、类别序号、视频序号

video_idx = -1
last_video_name = ''
cur_video_name = ''
for i, _ in  enumerate(sortedImgName_idx_target):
    imgName = sortedImgName_idx_target[i][0]
    cur_video_name = imgName[0:imgName.rfind('-')]
    if cur_video_name != last_video_name:
        video_idx += 1
    sortedImgName_idx_target[i].append(video_idx)
    last_video_name = cur_video_name



data_num = len(torch_imgs_and_targets)

def sample_and_build_relation(data_num, sample_num, target):
    relation_table = np.eye(data_num, dtype=np.int8)
    def generate_sample_idx():
        unknow_idx = np.where(relation_table == 0)
        unknow_relation_num = np.shape(unknow_idx[0])[0]
        if unknow_relation_num == 0:
            return None
        random_number = random.randint(0,unknow_relation_num-1) # 生成一个随机数，表示将要采集的索引
        return [unknow_idx[0][random_number], unknow_idx[1][random_number]]

    def update_relation_table(idx, relation):
        relation_table[idx[0], idx[1]] = relation
        relation_table[idx[1], idx[0]] = relation
        # 如果当前是正样本的话，首先这俩样本的正负样本需要统一，我的正样本也都是你的正样本，我的负样本也都是你的负样本，下面这几行做的就是这件事
        if relation == 1:
            for i in range(data_num):
                if relation_table[idx[1],i] != 0:
                    relation_table[idx[0],i] = relation_table[idx[1],i]
                if relation_table[idx[0],i] != 0:
                    relation_table[idx[1],i] = relation_table[idx[0],i]
        
        # 因为当前更新了索引i和j，所以它们所有的正样本也需要更新，并且更新完之后需要保证是对角阵
        pos_idx = np.where(relation_table[idx[0],:] == 1)
        relation_table[pos_idx,:] = relation_table[idx[0],:]
        relation_table[:,pos_idx] = relation_table[idx[0],:].reshape(data_num,1,1)

        pos_idx = np.where(relation_table[idx[1],:] == 1)
        relation_table[pos_idx,:] = relation_table[idx[1],:]
        relation_table[:,pos_idx] = relation_table[idx[1],:].reshape(data_num,1,1)

    pbar = tqdm(range(sample_num),desc='Sampling')
    for i in pbar:
        start = time.time()
        idx = generate_sample_idx()
        # print('generate sample idx: %.5f'%(time.time() - start))
        if idx == None:
            return relation_table
        if (target[idx[0]] == target[idx[1]]):
            relation = 1
        else:
            relation = -1
        start = time.time()
        update_relation_table(idx, relation)
        # print('update matrix time:  %.5f'%(time.time() - start))
    print('Sample done.')
    return relation_table

def check(relation_table, targets):
    print('Checking...')
    data_num = np.shape(relation_table)[0]
    for i in range(data_num):
        # 检查是否是对角阵以及关系是否对应标签
        for j in range(i, data_num):
            assert relation_table[i,j] == relation_table[j,i] # 检查是否是对角矩阵
            if relation_table[i,j] != 0:
                if targets[i] == targets[j]:
                    assert relation_table[i,j] == 1 # 根据标签检查关系是否正确
                else:
                    assert relation_table[i,j] == -1

        # 检查我的正样本和负样本们之间的关系是否正确
        pos = []
        neg = []
        for j in range(data_num):
            if relation_table[i,j] == 1: pos.append(j)
            if relation_table[i,j] == -1: neg.append(j)

        for j in pos: # 我的正样本们互为正样本
            for k in pos:
                assert relation_table[j,k] == 1
        for j in pos: # 我的正样本和负样本互为负样本
            for k in neg:
                assert relation_table[j,k] == -1
    print('Congratulations! You made it!!!!!')


num_for_sample = int(data_num/(same_range * 2 + 1))
sample_idx = list(range(same_range, data_num, same_range * 2 + 1)) # 这些索引用于采样
targets_of_sample = [sortedImgName_idx_target[i][2] for i in sample_idx] # 用于采样的数据的类别

relation_table = sample_and_build_relation(num_for_sample, sample_num, targets_of_sample)
# check(relation_table,targets_of_sample)


total_relation = num_for_sample*(num_for_sample-1)/2
left_unknown_relation = int(np.sum(relation_table == 0) / 2)
known_relation = total_relation - left_unknown_relation
print('num_for_sample: %d'%num_for_sample)
print('Sample number: %d'%sample_num)
print('Known relation: %d'%known_relation)
print('Total relation: %d'%total_relation)
print('Known relation percent: %.2f%%'%(100*known_relation/total_relation))

# 开始构建字典
print('Init dict...')
relation_dict = {}
for i in range(data_num):
    relation_dict[i] = {'pos':[],'neg':[]} # 初始化字典

for i in range(num_for_sample):
    range_pos = list(range((same_range*2+1)*i, (same_range*2+1)*(i+1)))
    for i in range_pos:
        for j in range_pos:
            if i == j or i >= data_num or j >= data_num: continue
            if sortedImgName_idx_target[i][3] == sortedImgName_idx_target[j][3]: # 如果这两帧来自同一个视频
                torch_i = sortedImgName_idx_target[i][1]
                torch_j = sortedImgName_idx_target[j][1]
                relation_dict[torch_i]['pos'].append(torch_j) # 首先，邻域内的样本互为正样本
print('Done')

pbar = tqdm(range(num_for_sample), desc='Building dict...')
for i in pbar:
    range1 = list(range((same_range*2+1)*i, (same_range*2+1)*(i+1)))
    for j in range(i+1, num_for_sample):
        relation = relation_table[i,j]
        if relation == 0: continue
        range2 = list(range((same_range*2+1)*j, (same_range*2+1)*(j+1)))
        for k in range1:
            for q in range2:
                if sortedImgName_idx_target[k][3] == sortedImgName_idx_target[(same_range*2+1)*i + same_range][3] and sortedImgName_idx_target[q][3] == sortedImgName_idx_target[(same_range*2+1)*j + same_range][3]:
                    torch_k = sortedImgName_idx_target[k][1]
                    torch_q = sortedImgName_idx_target[q][1]
                    if relation == 1: #torch_imgs_and_targets[torch_k][1] == torch_imgs_and_targets[torch_q][1]: 
                        relation_dict[torch_k]['pos'].append(torch_q)
                        relation_dict[torch_q]['pos'].append(torch_k)
                    if relation == -1: #torch_imgs_and_targets[torch_k][1] != torch_imgs_and_targets[torch_q][1]:
                        relation_dict[torch_k]['neg'].append(torch_q)
                        relation_dict[torch_q]['neg'].append(torch_k)
print('Done')

np.save(out_path, relation_dict)

# error_num = 0
# total = 0
# pbar = tqdm(relation_dict.items(), desc='Final checking')
# for key, value in pbar:
#     assert len(value['pos']) == len(set(value['pos']))
#     assert len(value['neg']) == len(set(value['neg']))
#     assert key not in value['pos']
#     assert key not in value['neg']
#     for i in value['pos']:
#         total += 1
#         assert key in relation_dict[i]['pos']
#         if torch_imgs_and_targets[i][1] != torch_imgs_and_targets[key][1]: error_num += 1
#     for i in value['neg']:
#         total += 1
#         assert key in relation_dict[i]['neg']
#         if torch_imgs_and_targets[i][1] == torch_imgs_and_targets[key][1]: error_num += 1



# print('Known relation: %d'%(total/2))
# print('Total relation: %d'%(data_num * (data_num - 1) / 2))
# print('Known relation percent: %.2f%%'%(100*total/(data_num * (data_num - 1))))
# print('Error relation: %d'%(error_num/2))
# print('Error rate: %.2f%%'%((100*error_num/total)/2))


print('Program exit normally.')