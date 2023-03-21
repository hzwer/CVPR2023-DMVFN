import os
import cv2
import sys
import math
import time
import torch
import lpips
import random
import logging
import argparse
import importlib
import numpy as np
import torch.distributed as dist
from pytorch_msssim import ssim, ms_ssim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler


root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from utils.util import *
from model.model import Model


def base_build_dataset(name):
    return getattr(importlib.import_module('dataset.dataset', package=None), name)()


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=300, type=int)
parser.add_argument('--num_gpu', default=4, type=int) # or 8
parser.add_argument('--batch_size', default=8, type=int, help='minibatch size')
parser.add_argument('--local_rank', default=0, type=int, help='local rank')
parser.add_argument('--train_dataset', required=True, type=str, help='CityTrainDataset, KittiTrainDataset, VimeoTrainDataset')
parser.add_argument('--val_datasets', type=str, nargs='+', default=['CityValDataset'], help='[CityValDataset,KittiValDataset,VimeoValDataset,DavisValDataset]')
parser.add_argument('--resume_path', default=None, type=str, help='continue to train, model path')
parser.add_argument('--resume_epoch', default=0, type=int, help='continue to train, epoch')
args = parser.parse_args()

torch.distributed.init_process_group(backend="nccl", world_size=args.num_gpu)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

exp = os.path.abspath('.').split('/')[-1]
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
log_path = './logs/train_log_{}/{}'.format(args.train_dataset, exp)
save_model_path = './models/train_log_{}/{}'.format(args.train_dataset, exp)


if local_rank == 0:
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    setup_logger('base', log_path, 'train', level=logging.INFO, screen=True, to_file=True)
    writer = SummaryWriter(log_path + '/train')
    writer_val = SummaryWriter(log_path + '/validate')

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
    return (1e-4 - 1e-5) * mul + 1e-5

logger = logging.getLogger('base')

def train(model, args):
    step = 0
    nr_eval = args.resume_epoch
    dataset = base_build_dataset(args.train_dataset)
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()

    step = 0 + args.step_per_epoch * args.resume_epoch
    
    if local_rank == 0:
        print('training...')
    time_stamp = time.time()
    for epoch in range(args.resume_epoch, args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255. #B,3,C,H,W
            
            learning_rate = get_learning_rate(step)

            loss_avg = model.train(data_gpu, learning_rate)
            
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/loss_l1', loss_avg, step)
                writer.flush()
            if local_rank == 0:
                logger.info('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_avg:{:.4e}'.format( \
                    epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, loss_avg))
            step += 1
        nr_eval += 1
        if nr_eval % 1 == 0:
            for dataset_name in args.val_datasets:
                val_dataset = base_build_dataset(dataset_name)
                val_data = DataLoader(val_dataset, batch_size=1, pin_memory=True, num_workers=1)
                evaluate(model, val_data, dataset_name, nr_eval, step)
        if local_rank <= 0:    
            model.save_model(save_model_path, epoch, local_rank)   
        dist.barrier()

def evaluate(model, val_data, name, nr_eval, step):
    if name == "CityValDataset" or name == "KittiValDataset" or name == "DavisValDataset":
        with torch.no_grad():
            lpips_score_mine, psnr_score_mine, msssim_score_mine, ssim_score_mine = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
            time_stamp = time.time()
            num = val_data.__len__()
            for i, data in enumerate(val_data):
                data_gpu, _ = data
                data_gpu = data_gpu.to(device, non_blocking=True) / 255.
                preds = model.eval(data_gpu, name)

                b,n,c,h,w = preds.shape
                assert b==1 and n==5

                gt, pred = data_gpu[0], preds[0]
                for j in range(5):
                    psnr = -10 * math.log10(torch.mean((gt[j+4] - pred[j]) * (gt[j+4] - pred[j])).cpu().data)
                    ssim_val = ssim( gt[j+4:j+5], pred[j:j+1], data_range=1.0, size_average=False) # return (N,)
                    ms_ssim_val = ms_ssim( gt[j+4:j+5], pred[j:j+1], data_range=1.0, size_average=False ) #(N,)
                    x, y = ((gt[j+4:j+5]-0.5)*2.0).clone(), ((pred[j:j+1]-0.5)*2.0).clone()
                    lpips_val = loss_fn_alex(x, y)

                    lpips_score_mine[j] += lpips_val
                    ssim_score_mine[j] += ssim_val
                    msssim_score_mine[j] += ms_ssim_val
                    psnr_score_mine[j] += psnr
                    
                    gt_1 = (gt[j+4:j+5].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                    pred_1 = (pred[j:j+1].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                    if i == 50 and local_rank == 0:
                            imgs = np.concatenate((gt_1[0], pred_1[0]), 1)[:, :, ::-1]
                            writer_val.add_image(name+str(j) + '/img', imgs.copy(), step, dataformats='HWC')
            eval_time_interval = time.time() - time_stamp

            if local_rank != 0:
                return
            psnr_score_mine, ssim_score_mine, msssim_score_mine, lpips_score_mine = psnr_score_mine/num, ssim_score_mine/num, msssim_score_mine/num, lpips_score_mine/num
            for i in range(5):
                logger.info('%d             '%(nr_eval)+name+'  psnr_%d     '%(i)+'%.4f'%(sum(psnr_score_mine[:(i+1)])/(i+1))+'  ssim_%d     '%(i)+'%.4f'%(sum(ssim_score_mine[:(i+1)])/(i+1))+'  ms_ssim_%d     '%(i)+
                '%.4f'%(sum(msssim_score_mine[:(i+1)])/(i+1))+'  lpips_%d     '%(i)+'%.4f'%(sum(lpips_score_mine[:(i+1)])/(i+1)))

                writer_val.add_scalar(name+' psnr_%d'%(i),  psnr_score_mine[i], step)
                writer_val.add_scalar(name+' ssim_%d'%(i),  ssim_score_mine[i], step)
                writer_val.add_scalar(name+' ms_ssim_%d'%(i),  msssim_score_mine[i], step)
                writer_val.add_scalar(name+' lpips_%d'%(i),  lpips_score_mine[i], step)
    elif name=="VimeoValDataset":
        with torch.no_grad():
            lpips_score_mine, ssim_score_mine, msssim_score_mine, psnr_score_mine   = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
            time_stamp = time.time()
            num = val_data.__len__()
            for i, data in enumerate(val_data):
                data_gpu, _ = data
                data_gpu = data_gpu.to(device, non_blocking=True) / 255.
                preds = model.eval(data_gpu, name)

                b,n,c,h,w = preds.shape
                assert b==1 and n==1

                gt, pred = data_gpu[0], preds[0]
                psnr = -10 * math.log10(torch.mean((gt[2] - pred[0]) * (gt[2] - pred[0])).cpu().data)
                ssim_val = ssim( gt[2:3], pred[0:1], data_range=1.0, size_average=False) # return (N,)
                ms_ssim_val = ms_ssim( gt[2:3], pred[0:1], data_range=1.0, size_average=False ) #(N,)
                x, y = ((gt[2:3]-0.5)*2.0).clone(), ((pred[0:1]-0.5)*2.0).clone()
                lpips_val = loss_fn_alex(x, y)

                lpips_score_mine[0] += lpips_val
                ssim_score_mine[0] += ssim_val
                msssim_score_mine[0] += ms_ssim_val
                psnr_score_mine[0] += psnr
                
                gt_1 = (gt[2:3].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                pred_1 = (pred[0:1].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                if i == 50 and local_rank == 0:
                        imgs = np.concatenate((gt_1[0], pred_1[0]), 1)[:, :, ::-1]
                        writer_val.add_image(name+str(0) + '/img', imgs.copy(), step, dataformats='HWC')
            eval_time_interval = time.time() - time_stamp

            if local_rank != 0:
                return
            psnr_score_mine, ssim_score_mine, msssim_score_mine, lpips_score_mine = psnr_score_mine/num, ssim_score_mine/num, msssim_score_mine/num, lpips_score_mine/num
            for i in range(1):
                logger.info('%d             '%(nr_eval)+name+'  psnr_%d     '%(i)+'%.4f'%(sum(psnr_score_mine[:(i+1)])/(i+1))+'  ssim_%d     '%(i)+'%.4f'%(sum(ssim_score_mine[:(i+1)])/(i+1))+'  ms_ssim_%d     '%(i)+
                '%.4f'%(sum(msssim_score_mine[:(i+1)])/(i+1))+'  lpips_%d     '%(i)+'%.4f'%(sum(lpips_score_mine[:(i+1)])/(i+1)))

                writer_val.add_scalar(name+' psnr_%d'%(i),  psnr_score_mine[i], step)
                writer_val.add_scalar(name+' ssim_%d'%(i),  ssim_score_mine[i], step)
                writer_val.add_scalar(name+' ms_ssim_%d'%(i),  msssim_score_mine[i], step)
                writer_val.add_scalar(name+' lpips_%d'%(i),  lpips_score_mine[i], step)
    
if __name__ == "__main__":    
    model = Model(local_rank=args.local_rank, resume_path=args.resume_path, resume_epoch=args.resume_epoch)
    train(model, args)
        
