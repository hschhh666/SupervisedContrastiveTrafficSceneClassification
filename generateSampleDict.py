import torch
from torchvision.datasets import ImageFolder
import numpy as np
import os
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--sample_num', type=int)
parser.add_argument('--dataset_path', type=str)
args = parser.parse_args()

dataset_path = args.dataset_path
sample_num = args.sample_num

outpath = os.path.join(dataset_path,'pos_neg_relation')
if not os.path.exists(outpath):
    os.makedirs(outpath)
out_path = os.path.join(outpath,'%d.npy'%sample_num)

dataset_path = os.path.join(dataset_path,'train')


torch_imageFolder = ImageFolder(dataset_path)
torch_imgs_and_targets = torch_imageFolder.imgs

data_num = len(torch_imgs_and_targets)
pair_list = []
for i in range(data_num - 1):
    for j in range(i+1, data_num):
        pair_list.append([i,j])

sampled_pair = random.sample(pair_list, sample_num)

relation_dict = {}
for i in range(data_num):
    relation_dict[i] = {'pos':[],'neg':[]} # 初始化字典

pbar = tqdm(sampled_pair)
for i in pbar:
    i1 = i[0]
    i2 = i[1]
    relation = (torch_imgs_and_targets[i1][1] == torch_imgs_and_targets[i2][1])
    if relation:
        relation_dict[i1]['pos'].append(i2)
        relation_dict[i2]['pos'].append(i1)
    else:
        relation_dict[i1]['neg'].append(i2)
        relation_dict[i2]['neg'].append(i1)

relation_dict['sample_num'] = sample_num
np.save(out_path, relation_dict)

readed_dict = np.load(out_path, allow_pickle = True).item()
pass
