"""

This code refers to CMC:https://github.com/HobbitLong/CMC/#contrastive-multiview-coding

Author: Shaochi Hu
"""
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse

import numpy as np

from torchvision import transforms
from dataset import myImageFolder

from myModel import myResnet50
from NCE.NCEAverage import NCEAverage, E2EAverage, SupConLoss
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from util import adjust_learning_rate, AverageMeter,print_running_time, Logger
from sampleIdx import RandomBatchSamplerWithPosAndNeg
from processFeature import process_feature
import tensorboard_logger as tb_logger



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # training parameters
    parser.add_argument('--print_freq', type=int, default=1, help='print every print_freq batchs')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save model checkpoint every save_freq epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=12, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--contrastMethod', type=str, default='membank',choices=['e2e', 'membank'], help='method of contrast, e2e or membank')

    # optimizer parameters
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')    

    # network parameters
    parser.add_argument('--pos_num', type=int, default=49) # positive sample number
    parser.add_argument('--neg_num', type=int, default=150) # negative sample number
    parser.add_argument('--nce_t', type=float, default=0.2) # temperature parameter
    parser.add_argument('--nce_m', type=float, default=0.9) # memory update rate
    parser.add_argument('--feat_dim', type=int, default=64, help='dim of feat for inner product') # dimension of network's output

    # specify folder
    parser.add_argument('--data', type=str, default=None, help='path to training data') # 训练数据文件夹，即锚点/正负样本文件夹
    parser.add_argument('--test_data_folder', type=str, default=None, help='path to testing data') # 测试数据文件夹，即所有视频帧的文件夹
    parser.add_argument('--running_save_path', type=str, default=None, help='path to save data')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')

    # 其他参数
    parser.add_argument('--comment_info', type=str, default='', help='Comment message, donot influence program')
    parser.add_argument('--load_img_to_memory', action='store_true', help='load all images into memory to speed up')
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    curTime = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())    
    args.model_name = '{}_PosNum_{}_NegNum_{}_lr_{}_decay_{}_bsz_{}_featDim_{}_{}'.format(curTime, args.pos_num, args.neg_num, args.learning_rate,
                                                                            args.weight_decay, args.batch_size, args.feat_dim, args.comment_info)

    # 路径创建与检查
    args.model_path = os.path.join(args.running_save_path, 'modelPath')
    args.log_txt_path = os.path.join(args.running_save_path, 'logPath')
    args.result_path = os.path.join(args.running_save_path, 'resultPath')
    args.tb_folder = os.path.join(args.running_save_path, 'tbPath')
    if (args.data is None) or (args.model_path is None)  or (args.log_txt_path is None) or (args.result_path is None) or (args.test_data_folder is None):
        raise ValueError('one or more of the folders is None: data | model_path | log_txt_path | result_path | test_data_folder')
    if not os.path.isdir(args.data):
        raise ValueError('data path not exist: {}'.format(args.data))

    args.model_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    if not os.path.isdir(args.log_txt_path):
        os.makedirs(args.log_txt_path)

    args.result_path = os.path.join(args.result_path, args.model_name)
    if not os.path.isdir(args.result_path):
        os.makedirs(args.result_path)
    

    args.tb_folder = os.path.join(args.tb_folder, args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)
    
    
    log_file_name = os.path.join(args.log_txt_path, 'log_'+args.model_name+'.txt') 
    sys.stdout = Logger(log_file_name) # 把print的东西输出到txt文件中

    for arg in vars(args):
        print(arg, ':', getattr(args, arg))  # getattr() 函数是获取args中arg的属性值
    
    print('start program at ' + time.strftime("%Y_%m_%d %H:%M:%S", time.localtime()))
    return args


def get_train_loader(args):
    data = os.path.join(args.data, 'train')

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    augmentation = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # transforms.GaussianBlur(9, (0.1,3)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = myImageFolder(data, transform=augmentation, memory = args.load_img_to_memory)
    args.n_data = len(train_dataset)
    print('number of train samples: {}'.format(args.n_data))
    args.dataset_targets = train_dataset.targets

    if args.contrastMethod == 'e2e':
        batch_sampler = RandomBatchSamplerWithPosAndNeg(train_dataset, args=args)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
    if args.contrastMethod == 'membank':
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    return train_loader

def set_model(args):

    model = myResnet50(args.feat_dim, pretrained=True)

    if args.resume:
        if torch.cuda.is_available():
            ckpt = torch.load(args.resume)
        else:
            ckpt = torch.load(args.resume,map_location=torch.device('cpu'))
        print("==> loaded pre-trained checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))
        model.load_state_dict(ckpt['model'])
        print('==> done')

    if args.contrastMethod == 'membank':
        criterion = SupConLoss(args.n_data,args.dataset_targets,args.pos_num, args.neg_num, args.feat_dim, args.nce_t, args.nce_m)

    elif args.contrastMethod == 'e2e':
        contrast = E2EAverage(args.nce_k, args.n_data, args.nce_t, args.softmax)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer

def train_e2e(epoch,train_loader, model, contrast, criterion, optimizer, args):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    probs = AverageMeter()

    end = time.time()
    for idx,(img, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = img.size(0)
        if torch.cuda.is_available():
            img = img.cuda()

        # ===================forward=====================
        feat = model(img)
        loss = criterion(feat,target, index)
        mutualInfo = contrast(feat)
        loss = criterion(mutualInfo)
        prob = mutualInfo[:,0].mean()
        prob = 1/torch.exp(loss)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        probs.update(prob.item(), bsz)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'p {probs.val:.3f} ({probs.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, probs=probs,))
            sys.stdout.flush()

    return losses.avg, probs.avg



def train_mem_bank(epoch,train_loader, model, criterion, optimizer, args):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx,(img, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = img.size(0)
        img = img.float()
        if torch.cuda.is_available():
            index = index.cuda()
            img = img.cuda()

        # ===================forward=====================
        feat = model(img)
        loss = criterion(feat, target, index)


        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  .format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, ))
            sys.stdout.flush()

    return losses.avg


def main():
    # parse the args
    args = parse_option()
    args.start_epoch = 1

    # set the loader
    train_loader = get_train_loader(args)

    # set the model
    model, criterion = set_model(args)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # train by epoch
    print('start training at ' + time.strftime("%Y_%m_%d %H:%M:%S", time.localtime()))
    start_time = time.time()
    min_loss = np.inf
    best_model_path = ''
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(epoch, args, optimizer)

        if args.contrastMethod == 'e2e':
            loss, prob = train_e2e(epoch, train_loader, model, criterion, optimizer, args)
        else:
            loss = train_mem_bank(epoch, train_loader, model, criterion, optimizer, args)

        print_running_time(start_time)

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            # if args.amp:
            #     state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        if loss < min_loss:
            if min_loss != np.inf:
                os.remove(best_model_path)
            min_loss = loss
            best_model_path = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}_Best.pth'.format(epoch=epoch))
            print('==> Saving best model...')
            state = {
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            # if args.amp:
            #     state['amp'] = amp.state_dict()
            torch.save(state, best_model_path)
            # help release GPU memory
            del state
    
    print("==================== Training finished. Start testing ====================")
    print('==> loading best model')
    print('min loss = %.3f'%min_loss)
    args.pretrained = best_model_path
    
    process_feature(args)

    print('Program exit normally.')

if __name__ == '__main__':
    main()


