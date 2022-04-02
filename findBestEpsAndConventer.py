from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from torchvision.datasets import ImageFolder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy
import os
import sys

from util import Logger,print_running_time



dataset_path = '/home/hsc/Research/TrafficSceneClassification/data/fineGrain/dataset5'
feat_path = '/home/hsc/Research/TrafficSceneClassification/runningSavePathSupCon/resultPath/20220402_17_32_55_PosNum_1200_NegNum_1200_lr_0.03_decay_0.0001_bsz_128_featDim_64_testCode'

log_file_name = os.path.join(feat_path, 'clusterAndEvaluation.txt') 
sys.stdout = Logger(log_file_name) # 把print的东西输出到txt文件中

train_data_path = os.path.join(dataset_path,'train')
val_data_path = os.path.join(dataset_path,'test')

train_targets = ImageFolder(train_data_path).targets
val_targets = ImageFolder(val_data_path).targets

train_imgfolder = ImageFolder(train_data_path)
val_imgfolder = ImageFolder(val_data_path)

train_feat = os.path.join(feat_path,'train_feat_after_fc.npy')
val_feat = os.path.join(feat_path,'val_feat_after_fc.npy')
train_feat = np.load(train_feat)
val_feat = np.load(val_feat)
train_feat = train_feat / np.linalg.norm(train_feat,axis=1, keepdims=True) # 撒币了，在计算特征的时候忘了归一化了
val_feat = val_feat / np.linalg.norm(val_feat,axis=1, keepdims=True) # 撒币了，在计算特征的时候忘了归一化了

#==========================================
eps = 0.3
colars = ['red',(112/255,173/255,71/255),(149/255,72/255,162/255),(2/255,176/255,240/255)]


# cluster_model = DBSCAN(eps=eps,min_samples=50, metric='cosine').fit(train_feat)
# cluster_label = cluster_model.labels_

cluster_model = KMeans(n_clusters=4).fit(train_feat)
cluster_label = cluster_model.labels_

for i in range(len(set(cluster_label))):
    print(i, np.shape(cluster_label[cluster_label == i])[0])

cluster_num = len(set(cluster_label))
if -1 in cluster_label:
    cluster_num -= 1

centers = []
for i in range(cluster_num):
    cur_center = train_feat[cluster_label == i]
    cur_center = np.average(cur_center,axis=0)
    centers.append(cur_center)

centers = np.array(centers)
pca_model = PCA(n_components=2).fit(train_feat)
reduced_train_feat_pca = pca_model.transform(train_feat)
reduced_center_feat = pca_model.transform(centers)


fig = plt.figure()
fig.add_subplot(111)
plt.scatter(reduced_train_feat_pca[:,0], reduced_train_feat_pca[:,1],c = train_targets, s = 1, alpha = 0.5)
plt.scatter(reduced_center_feat[:,0], reduced_center_feat[:,1],c = 'black', s = 10, alpha = 1)
plt.xticks([])  #去掉横坐标值
plt.yticks([])  #去掉纵坐标值
plt.title('Training feat vis by PCA, eps %.2f, cluster num %d'%(eps, cluster_num))
plt.savefig(os.path.join(feat_path, 'train_cluster_pca.png'))
print('cluster num: ',cluster_num)


if not os.path.exists(os.path.join(feat_path, 'train_cluster_tsne.png')):
    reduced_train_feat_tsne = TSNE(n_components=2).fit_transform(train_feat)
    fig = plt.figure()
    fig.add_subplot(111)
    plt.scatter(reduced_train_feat_tsne[:,0], reduced_train_feat_tsne[:,1],c = train_targets, s = 1, alpha = 0.5)
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值
    plt.title('Training feat vis by tsne, eps %.2f, cluster num %d'%(eps, cluster_num))
    plt.savefig(os.path.join(feat_path, 'train_cluster_tsne.png'))


max_acc = 0
good_conventer = []

label_conventers = itertools.permutations([0,1,2,3])
for label_conventer in label_conventers:
    print('label conventer:',label_conventer)
    converted_label = np.zeros_like(cluster_label)
    for i in range(len(cluster_label)):
        converted_label[i] = label_conventer[cluster_label[i]]

    converted_center = [0,0,0,0]
    for i in range(cluster_num):
        converted_center[label_conventer[i]] = np.average(train_feat[cluster_label == i], axis = 0)

    converted_center = np.array(converted_center) # 这就是聚类中心
    converted_center = converted_center / np.linalg.norm(converted_center,axis=1, keepdims=True) # 撒币了，这里需要重新归一化的

    pred_labels = []
    for i in range(np.shape(val_feat)[0]):
        cur_feat = val_feat[i]
        min_dis = np.inf
        p = 0
        for j in range(np.shape(converted_center)[0]):
            cur_center = converted_center[j]
            dis = 1 - np.sum((cur_center/np.linalg.norm(cur_center)) * (cur_feat/np.linalg.norm(cur_feat)))
            if dis < min_dis:
                min_dis = dis
                p = j
        pred_labels.append(p)
    
    acc = accuracy_score(val_targets, pred_labels)
    if acc > max_acc:
        max_acc = acc
        good_conventer = copy.deepcopy(label_conventer)
    

    print('<================Val classification report================>')
    print(classification_report(val_targets, pred_labels, target_names=['Highway','Local','Ramp','Urban'], digits=3))
    print('<================Val classification report================>')




label_conventer = good_conventer
converted_label = np.zeros_like(cluster_label)
for i in range(len(cluster_label)):
    converted_label[i] = label_conventer[cluster_label[i]]

converted_center = [0,0,0,0]
for i in range(cluster_num):
    converted_center[label_conventer[i]] = np.average(train_feat[cluster_label == i], axis = 0)

converted_center = np.array(converted_center) # 这就是聚类中心
converted_center = converted_center / np.linalg.norm(converted_center,axis=1, keepdims=True) # 撒币了，这里需要重新归一化的


pred_labels = []
for i in range(np.shape(val_feat)[0]):
    cur_feat = val_feat[i]
    min_dis = np.inf
    p = 0
    for j in range(np.shape(converted_center)[0]):
        cur_center = converted_center[j]
        dis = 1 - np.sum((cur_center/np.linalg.norm(cur_center)) * (cur_feat/np.linalg.norm(cur_feat)))
        if dis < min_dis:
            min_dis = dis
            p = j
    pred_labels.append(p)


np.save(os.path.join(feat_path, 'pred.npy'), np.array(pred_labels, dtype=int))
cm = confusion_matrix(val_targets, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Highway','Local','Ramp','Urban'])
disp.plot()
plt.title('Val')
plt.savefig(os.path.join(feat_path,'confusion_matrix.png'))
print('===================================================================================\n\n')
print('Best conventer: ', good_conventer)
print('<================Best classification report================>')
print(classification_report(val_targets, pred_labels, target_names=['Highway','Local','Ramp','Urban'], digits=3))
print('<================Best classification report================>')




reduced_val_feat_pca = pca_model.transform(val_feat)
fig = plt.figure(figsize=(6, 6), dpi=600)
fig.add_subplot(111)
plt.scatter(reduced_val_feat_pca[:,0], reduced_val_feat_pca[:,1],c = [colars[i] for i in val_targets], s = 10, alpha = 0.1, zorder=1)
plt.xticks([])  #去掉横坐标值
plt.yticks([])  #去掉纵坐标值
plt.axis('off')  # 去掉坐标轴
plt.savefig(os.path.join(feat_path, 'test_feat_on_train_pca.png'), bbox_inches='tight', pad_inches=0)


reduced_val_feat_tsne = TSNE(n_components=2).fit_transform(val_feat)
fig = plt.figure(figsize=(6, 6), dpi=600)
fig.add_subplot(111)
plt.scatter(reduced_val_feat_tsne[:,0], reduced_val_feat_tsne[:,1],c = [colars[i] for i in val_targets],s = 5, alpha = 0.5, zorder=1)
plt.xticks([])  #去掉横坐标值
plt.yticks([])  #去掉纵坐标值
plt.axis('off')  # 去掉坐标轴
plt.savefig(os.path.join(feat_path, 'test_feat_on_train_tsne.png'), bbox_inches='tight', pad_inches=0)