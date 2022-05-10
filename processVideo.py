import argparse
import math
import os
from pydoc import classname
import sys
import random
import shutil
import time
import warnings
import numpy as np
import cv2
from tqdm import tqdm

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
from torchvision.transforms.transforms import Resize
from util import Logger,print_running_time, print_time
from myModel import myResnet50
from PIL import Image


@print_time
def process_video(args):
    if args.video_path is None: return 

    video_path = args.video_path

    mp4_file = []
    
    for _,_,file in os.walk(video_path):
        for i in file:
            if i.endswith('mp4'):
                mp4_file.append(os.path.join(video_path,i))
        break

    # mp4_file = [1]
    for v in mp4_file:
        args.video = v
        # args.video = '/home/hsc/Research/TrafficSceneClassification/data/video/finGrainTrain/201803261504_2018-03-26.mp4'

        
        args.resultPath = args.pretrained.replace('modelPath','resultPath')
        args.resultPath = '/'.join(args.resultPath.split('/')[:-1])
        args.tarNpyPath = os.path.join(args.resultPath, 'video_feat')
        if not os.path.exists(args.tarNpyPath):
            os.makedirs(args.tarNpyPath)


        args.tarNpyPath =os.path.join(args.tarNpyPath, args.video.split('/')[-1][:-4])
        # ===============读取视频及信息===============
        cap = cv2.VideoCapture(args.video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



        # ===============配置特征计算模型===============
        model = myResnet50(64, parallel = False)
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        print('=> checkpoint epoch {}'.format(checkpoint['epoch']))
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            state_dict[k.replace('module.','')] = state_dict[k]
            del state_dict[k]
        print("=> loaded pre-trained model '{}'".format(args.pretrained))
        msg = model.load_state_dict(state_dict, strict=1)
        model.cuda()
        model.eval()


        # ===============特征计算的transforms===============  
        featCal_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ])


        # ===============中间层特征提取===============
        class FeatureExtractor(nn.Module):
            def __init__(self, submodule):
                super(FeatureExtractor, self).__init__()
                self.submodule = submodule
        
            # 自己修改forward函数
            def forward(self, x):
                res = []
                for name, module in self.submodule._modules['model']._modules.items():
                    if name == "fc": 
                        x = x.view(x.size(0), -1)
                        res.append(x)
                    x = module(x)
                res.append(x)
                return res

        feature_extractor = FeatureExtractor(model)
        memory_after_fc = torch.ones(frame_num, model.feat_dim_after_fc).cuda()

        # ==============================开始处理视频==============================  
        print(args.video)
        pbar = tqdm(range(frame_num))
        for i in pbar:
            fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, img = cap.read()
            if not ret:
                print('Read error at frame %d in %s'%(fno, args.video))
                continue
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            img = cv2.resize(img,(400,224))
            
            img = Image.fromarray(img).convert('RGB')
            
            img = featCal_transforms(img)
            img = img.unsqueeze(0)
            img = img.cuda()
            with torch.no_grad():
                res = feature_extractor(img)
            feat_after_fc = res[1]
            index = [fno]
            index = torch.tensor(index, dtype=torch.long).cuda()
            memory_after_fc.index_copy_(0,index,feat_after_fc)


        memory_after_fc = memory_after_fc.cpu().numpy()
        np.save(args.tarNpyPath+'_memoryFeature.npy', memory_after_fc)

        cap.release()
        print('Save feature npy file to %s. Done.'%(args.tarNpyPath+'_memoryFeature.npy'))



        #=====================根据真值计算训练集类别中心===========================

        train_feat = os.path.join(args.resultPath,'train_feat_after_fc.npy')
        train_targets = np.load(os.path.join(args.resultPath,'train_targets.npy'))
        train_feat = np.load(train_feat)
        train_feat = train_feat / np.linalg.norm(train_feat,axis=1, keepdims=True) # 撒币了，在计算特征的时候忘了归一化了
        memory_after_fc = memory_after_fc / np.linalg.norm(memory_after_fc,axis=1, keepdims=True) # 撒币了，在计算特征的时候忘了归一化了

        class_num = len(set(train_targets))
        centers = []
        for i in range(class_num):
            cur_center = train_feat[train_targets == i]
            cur_center = np.average(cur_center,axis=0)
            centers.append(cur_center)

        centers = np.array(centers)
        centers = centers / np.linalg.norm(centers,axis=1, keepdims=True) # 撒币了，这里需要重新归一化的


        pred_labels = []
        risks = []
        for i in range(np.shape(memory_after_fc)[0]):
            cur_feat = memory_after_fc[i]
            min_dis = np.inf
            p = 0
            cur_risk = []
            for j in range(np.shape(centers)[0]):
                cur_center = centers[j]
                dis = 1 - np.sum((cur_center/np.linalg.norm(cur_center)) * (cur_feat/np.linalg.norm(cur_feat)))
                cur_risk.append(dis)
                if dis < min_dis:
                    min_dis = dis
                    p = j
            risks.append(cur_risk)
            pred_labels.append(p)

        risks = np.array(risks)
        pred_labels = np.array(pred_labels, dtype=int)
        np.save(args.tarNpyPath+'_risks.npy', risks)
        np.save(args.tarNpyPath+'_predLabels.npy', pred_labels)
        print('Save feature npy file to %s. Done.'%(args.tarNpyPath+'_risks.npy'))
        print('Save feature npy file to %s. Done.'%(args.tarNpyPath+'_predLabels.npy'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.pretrained = '/home/hsc/Research/TrafficSceneClassification/runningSavePathSupCon/modelPath/20220427_21_04_29_PosNum_30_NegNum_1200_lr_0.03_decay_0.0001_bsz_128_featDim_64_dataset5witSmallClass/ckpt_epoch_180_Best.pth'
    # args.pretrained = '/home/hsc/Research/TrafficSceneClassification/runningSavePathSupCon/modelPath/20220402_04_29_07_PosNum_1200_NegNum_1200_lr_0.03_decay_0.0001_bsz_128_featDim_64_/ckpt_epoch_150_Best.pth'

    args.video_path = '/home/hsc/Research/TrafficSceneClassification/data/video/tmp'
    process_video(args)