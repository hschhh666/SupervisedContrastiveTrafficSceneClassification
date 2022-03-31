import argparse
import math
import os
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
from util import Logger,print_running_time
from myModel import myResnet50
from PIL import Image

sys.path.append('/home/hsc/Research/TrafficSceneClassification/code/testExperiment/Deeplab/deeplabv3')
from model.deeplabv3 import DeepLabV3


parser = argparse.ArgumentParser()
parser.add_argument('--video', default='')
args = parser.parse_args()
args.pretrained = '/home/hsc/Research/TrafficSceneClassification/runningSavePath/modelPath/20211211_04_03_50_lossMethod_softmax_NegNum_128_lr_0.03_decay_0.0001_bsz_128_featDim_64_/ckpt_epoch_386_Best.pth'


# args.video = '/home/hsc/Research/TrafficSceneClassification/data/val/20180405_113845_2018-04-05.mp4'

args.tarVideo = args.video[:-4]+'_masked.avi'
args.tarNpyPath = args.video[:-4]+'_Contras.npy'
# ===============读取视频及信息===============
cap = cv2.VideoCapture(args.video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if os.path.exists(args.tarVideo):
    cap.release()
    cap = cv2.VideoCapture(args.tarVideo)
    print('Masked video exist, donot calculate mask.')
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.tarVideo, fourcc, fps, (width, height))


# ===============配置语义分割模型===============
if not os.path.exists(args.tarVideo):
    semantic_model = DeepLabV3()
    semantic_model.load_state_dict(torch.load("/home/hsc/Research/TrafficSceneClassification/code/testExperiment/Deeplab/deeplabv3/pretrained_models/model_13_2_2_2_epoch_580.pth"))
    semantic_model.cuda()
    semantic_model.eval()

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

# ===============语义分割的transforms===============
semantic_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

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
pbar = tqdm(range(frame_num))
for i in pbar:
    fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, img = cap.read()
    if not ret:
        print('Read error at frame %d in %s'%(fno, args.video))
        continue
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    if not os.path.exists(args.tarVideo):
        img = Image.fromarray(img).convert('RGB')
        img = semantic_transforms(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            outputs = semantic_model(img)
        outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
        pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)
        img = img.cpu().numpy()[0]
        pred_label_imgs = np.transpose(pred_label_imgs,(1,2,0))[:,:,0]
        img = np.transpose(img,(1,2,0))
        img = img*np.array([0.229, 0.224, 0.225])
        img = img + np.array([0.485, 0.456, 0.406])
        img = img*255.0
        img = img.astype(np.uint8)
        labels = list(range(11,19))
        mask = np.zeros((np.shape(img)[0], np.shape(img)[1]), dtype= int)
        for l in labels:
            mask = np.logical_or(mask, pred_label_imgs == l)
        mask = 1 - mask
        img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask = mask.astype(np.uint8))
        #至此，semantic处理完毕一帧图像
        frame = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        out.write(frame)


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
np.save(args.tarNpyPath, memory_after_fc)
if not os.path.exists(args.tarVideo):
    out.release()
cap.release()
print('Save npy to %s. Done.'%args.tarNpyPath)