import os
import sys
import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from util import Logger, print_time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse

@print_time
def evaluation(args):
    # 指定路径
    dataset_path = args.data
    feat_path = args.result_path

    # 指定降维后可视化各类别的颜色
    colars = ['red',(112/255,173/255,71/255),(149/255,72/255,162/255),(2/255,176/255,240/255),'green','blue','orange']

    # 加载数据
    log_file_name = os.path.join(feat_path, 'clusterAndEvaluation.txt') 
    sys.stdout = Logger(log_file_name) # 把print的东西输出到txt文件中
    train_feat = os.path.join(feat_path,'train_feat_after_fc.npy')
    train_targets = os.path.join(feat_path,'train_targets.npy')
    val_feat = os.path.join(feat_path,'val_feat_after_fc.npy')
    val_targets = os.path.join(feat_path,'val_targets.npy')
    train_feat = np.load(train_feat)
    train_targets = np.load(train_targets)
    val_feat = np.load(val_feat)
    val_targets = np.load(val_targets)
    train_feat = train_feat / np.linalg.norm(train_feat,axis=1, keepdims=True) # 撒币了，在计算特征的时候忘了归一化了
    val_feat = val_feat / np.linalg.norm(val_feat,axis=1, keepdims=True) # 撒币了，在计算特征的时候忘了归一化了


    # 获取类别名称和数量
    train_data_path = os.path.join(dataset_path,'train')
    train_imgfolder = ImageFolder(train_data_path)
    class_name = train_imgfolder.classes
    class_num = len(class_name)
    print('Training data size:')
    for i in range(class_num):
        c = class_name[i]
        n = np.shape(train_targets[train_targets==i])[0]
        print('%-10s: %d'%(c,n))
    print()

    # 计算类别中心
    centers = []
    for i in range(class_num):
        cur_center = train_feat[train_targets == i]
        cur_center = np.average(cur_center,axis=0)
        centers.append(cur_center)
    centers = np.array(centers)



    # 预测测试集标签
    centers = centers / np.linalg.norm(centers,axis=1, keepdims=True) # 撒币了，这里需要重新归一化的
    pred_labels = []
    for i in range(np.shape(val_feat)[0]):
        cur_feat = val_feat[i]
        min_dis = np.inf
        p = 0
        for j in range(np.shape(centers)[0]):
            cur_center = centers[j]
            dis = 1 - np.sum((cur_center/np.linalg.norm(cur_center)) * (cur_feat/np.linalg.norm(cur_feat)))
            if dis < min_dis:
                min_dis = dis
                p = j
        pred_labels.append(p)

    # 保存预测值，输出预测的混淆矩阵
    np.save(os.path.join(feat_path, 'pred.npy'), np.array(pred_labels, dtype=int))
    cm = confusion_matrix(val_targets, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_name)
    disp.plot()
    plt.title('Test')
    plt.savefig(os.path.join(feat_path,'confusion_matrix.png'))
    print('<================Best classification report================>')
    print(classification_report(val_targets, pred_labels, target_names=class_name, digits=3))
    print('<================Best classification report================>')
    print()



    # 计算训练集类间平均相似度
    print('Inter class cosine distance in training set')
    for i in range(class_num):
        c1 = class_name[i]
        feat = train_feat[train_targets == i, :]
        for j in range(class_num):
            c2 = class_name[j]
            feat2 = train_feat[train_targets == j, :]
            v = 1-np.mean(np.matmul(feat, feat2.T))
            print('%.3f'%v, end=' ')
        print()
    print()

    # 计算测试集类间平均相似度
    print('Inter class cosine distance in testing set')
    for i in range(class_num):
        c1 = class_name[i]
        feat = val_feat[val_targets == i, :]
        for j in range(class_num):
            c2 = class_name[j]
            feat2 = val_feat[val_targets == j, :]
            v = 1-np.mean(np.matmul(feat, feat2.T))
            print('%.3f'%v, end=' ')
        print()
    print()


    # 可视化训练数据PCA降维后的图像，包含类别中心
    pca_model = PCA(n_components=2).fit(train_feat)
    reduced_train_feat_pca = pca_model.transform(train_feat)
    reduced_center_feat = pca_model.transform(centers)
    fig = plt.figure(figsize=(6, 6), dpi=600)
    fig.add_subplot(111)
    c = np.array([colars[i] for i in train_targets], dtype=object)
    plt.scatter(reduced_train_feat_pca[:,0], reduced_train_feat_pca[:,1],c = c, s = 1, alpha = 0.5)
    plt.scatter(reduced_center_feat[:,0], reduced_center_feat[:,1],c = 'black', s = 10, alpha = 1)
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值
    plt.axis('off')  # 去掉坐标轴
    plt.savefig(os.path.join(feat_path, 'train_pca.png'), bbox_inches='tight', pad_inches=0)


    # 可视化训练数据TSNE降维后的图像
    reduced_train_feat_tsne = TSNE(n_components=2).fit_transform(train_feat)
    fig = plt.figure(figsize=(6, 6), dpi=600)
    fig.add_subplot(111)
    c = np.array([colars[i] for i in train_targets], dtype=object)
    plt.scatter(reduced_train_feat_tsne[:,0], reduced_train_feat_tsne[:,1],c = c, s = 1, alpha = 0.5)
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值
    plt.axis('off')  # 去掉坐标轴
    plt.savefig(os.path.join(feat_path, 'train_tsne.png'), bbox_inches='tight', pad_inches=0)


    # 可视化测试数据PCA降维后的图像
    reduced_val_feat_pca = pca_model.transform(val_feat)
    fig = plt.figure(figsize=(6, 6), dpi=600)
    fig.add_subplot(111)
    c = np.array([colars[i] for i in val_targets], dtype=object)
    plt.scatter(reduced_val_feat_pca[:,0], reduced_val_feat_pca[:,1],c = c, s = 10, alpha = 0.1, zorder=1)
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值
    plt.axis('off')  # 去掉坐标轴
    plt.savefig(os.path.join(feat_path, 'test_pca.png'), bbox_inches='tight', pad_inches=0)

    # 可视化测试数据TSNE降维后的图像
    reduced_val_feat_tsne = TSNE(n_components=2).fit_transform(val_feat)
    fig = plt.figure(figsize=(6, 6), dpi=600)
    fig.add_subplot(111)
    c = np.array([colars[i] for i in val_targets], dtype=object)
    plt.scatter(reduced_val_feat_tsne[:,0], reduced_val_feat_tsne[:,1],c = c,s = 5, alpha = 0.5, zorder=1)
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值
    plt.axis('off')  # 去掉坐标轴
    plt.savefig(os.path.join(feat_path, 'test_tsne.png'), bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data = '/home/hsc/Research/TrafficSceneClassification/data/fineGrain/dataset7'
    args.result_path = '/home/hsc/Research/TrafficSceneClassification/runningSavePathSupCon/resultPath/20220510_05_27_26_PosNum_30_NegNum_1200_lr_0.03_decay_0.0001_bsz_128_featDim_64_dataset7'

    evaluation(args)