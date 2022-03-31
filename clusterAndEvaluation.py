import argparse
import math
import os
import sys
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from util import Logger,print_running_time
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

res_path = '/home/hsc/Research/TrafficSceneClassification/runningSavePath/resultPath/20211205_02_24_08_lossMethod_softmax_NegNum_128_lr_0.03_decay_0.0001_bsz_128_featDim_64_'

log_file_name = os.path.join(res_path, 'clusterAndEvaluation.txt') 
sys.stdout = Logger(log_file_name) # 把print的东西输出到txt文件中


feat = os.path.join(res_path,'train_feat_after_fc.npy')
targets = os.path.join(res_path,'train_targets.npy')

val_feat = os.path.join(res_path,'val_feat_after_fc.npy')
val_targets = os.path.join(res_path,'val_targets.npy')

distance_metric = 'cosine'
# distance_metric = 'euclid'

feat = np.load(feat)
targets = np.load(targets)
val_feat = np.load(val_feat)
val_targets = np.load(val_targets)

cluster_model = KMeans(n_clusters=4).fit(feat)
cluster_label = cluster_model.labels_


map_table = np.zeros((4,4), dtype=int)
for i in range(len(cluster_label)):
    pre_label = cluster_label[i]
    gt_label = targets[i]
    map_table[pre_label, gt_label] += 1

label_conventer = np.zeros(4,dtype=int)
for i in range(4):
    label_conventer[i] = int(map_table[i,:].argmax())
    print(map_table[i,int(map_table[i,:].argmax())])

# label_conventer = [0,1,2,3]

converted_label = np.zeros_like(cluster_label)
for i in range(len(cluster_label)):
    converted_label[i] = label_conventer[cluster_label[i]]

converted_center = [0,0,0,0]
for i in range(np.shape(cluster_model.cluster_centers_)[0]):
    converted_center[label_conventer[i]] = cluster_model.cluster_centers_[i,:]

converted_center = np.array(converted_center) # 这就是聚类中心

# for i in range(4):
#     converted_center[i,:] = np.average(feat[targets == i], axis=0)


pred_labels = []
for i in range(np.shape(val_feat)[0]):
    cur_feat = val_feat[i]
    min_dis = np.inf
    p = 0
    for j in range(np.shape(converted_center)[0]):
        cur_center = converted_center[j]
        if distance_metric == 'euclid':
            dis = np.sqrt(np.sum(np.square(cur_center - cur_feat)))
        else:
            dis = 1 - np.sum((cur_center/np.linalg.norm(cur_center)) * (cur_feat/np.linalg.norm(cur_feat)))

        if dis < min_dis:
            min_dis = dis
            p = j
    pred_labels.append(p)

np.save(os.path.join(res_path, 'pred.npy'), np.array(pred_labels, dtype=int))

cm = confusion_matrix(val_targets, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Highway','Local','Ramp','Urban'])
disp.plot()
plt.savefig(os.path.join(res_path,'confusion_matrix.png'))

print('<================Val classification report================>')
print(classification_report(val_targets, pred_labels, target_names=['Highway','Local','Ramp','Urban'], digits=3))
print('<================Val classification report================>')

vis_reduce_method = 'PCA' # tSNE

if vis_reduce_method.lower() == 'pca':
    reduce_model = PCA(n_components=2)
else:
    reduce_model = TSNE()

feat = np.concatenate((feat, converted_center))
tmp_feat = reduce_model.fit_transform(feat)
reduced_feat = tmp_feat[:-4,:]
converted_center = tmp_feat[-4:,:]
fig = plt.figure()
fig.add_subplot(121)
plt.scatter(reduced_feat[:,0], reduced_feat[:,1], c = targets, s = 1, alpha = 0.8)
plt.title('gt-%s'%vis_reduce_method)
fig.add_subplot(122)
plt.scatter(converted_center[:,0], converted_center[:,1], c = [0,1,2,3], s = 40)
plt.scatter(reduced_feat[:,0], reduced_feat[:,1], c = converted_label, s = 1, alpha = 0.8)
plt.title('kmeans-%s'%vis_reduce_method)
plt.savefig(os.path.join(res_path,'cluster.png'))

exit(0)



