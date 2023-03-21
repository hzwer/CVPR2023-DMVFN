import os
import cv2
import sys
import math
import time
import torch
import random
import argparse
import lpips
import logging
import importlib
import numpy as np
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim
from torch.utils.data import DataLoader, Dataset


root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from utils.util import *
from model.model import Model

device = torch.device("cuda")
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

def base_build_dataset(name):
    return getattr(importlib.import_module('dataset.dataset', package=None), name)()

parser = argparse.ArgumentParser()
# parser.add_argument('--local_rank', default=0, type=int, help='local rank')
parser.add_argument('--save_img', action='store_true', help='save or not')
parser.add_argument('--val_datasets', type=str, nargs='+', default=['CityValDataset'], help='[CityValDataset,KittiValDataset,VimeoValDataset,DavisValDataset]')
parser.add_argument('--load_path', required=True, type=str, help='model path')
args = parser.parse_args()

exp = os.path.abspath('.').split('/')[-1]
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
log_path = './logs/test_log/{}'.format(exp)


if not os.path.exists(log_path):
    os.makedirs(log_path)

setup_logger('base', log_path, 'test', level=logging.INFO, screen=True, to_file=True)

logger = logging.getLogger('base')

def test(model, save_img=False):
    step = 0
    nr_eval = 0
    print('Start testing...')


    for dataset_name in args.val_datasets:
        val_dataset = base_build_dataset(dataset_name)
        val_data_city = DataLoader(val_dataset, batch_size=1, pin_memory=True, num_workers=1)
        evaluate(model, val_data_city, dataset_name, nr_eval, step, save_img) 


def evaluate(model, val_data, name, nr_eval, step, save_img):
    
    save_img_path = './save_img/test_log_{}/{}'.format(name, exp)
    if name == "CityValDataset" or name == "KittiValDataset" or name == "DavisValDataset":
        with torch.no_grad():
            lpips_score_mine, psnr_score_mine, msssim_score_mine, ssim_score_mine = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)

            time_stamp = time.time()
            num = val_data.__len__()
            print("Dataset Length:", num)

            for i, data in tqdm(enumerate(val_data), desc="Processing", total=num):
                data_gpu, data_name = data
                data_gpu = data_gpu.to(device, non_blocking=True) / 255.
                preds = model.eval(data_gpu, name)

                b,n,c,h,w = preds.shape
                assert b==1 and n==5

                gt, pred = data_gpu[0], preds[0]
                if save_img:
                    pred_1_name = os.path.join(save_img_path, data_name[0])
                    print(pred_1_name, data_name)
                    if not os.path.exists(pred_1_name):
                        os.makedirs(pred_1_name)

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

                    if save_img:
                        cv2.imwrite(os.path.join(pred_1_name, 'pred_%d.png'%(j+1)), pred_1[0])

            eval_time_interval = time.time() - time_stamp

            psnr_score_mine, ssim_score_mine, msssim_score_mine, lpips_score_mine = psnr_score_mine/num, ssim_score_mine/num, msssim_score_mine/num, lpips_score_mine/num
            for i in range(5):
                logger.info('%d             '%(nr_eval)+name+'  psnr_%d     '%(i)+'%.4f'%(sum(psnr_score_mine[:(i+1)])/(i+1))+'  ssim_%d     '%(i)+'%.4f'%(sum(ssim_score_mine[:(i+1)])/(i+1))+'  ms_ssim_%d     '%(i)+
                '%.4f'%(sum(msssim_score_mine[:(i+1)])/(i+1))+'  lpips_%d     '%(i)+'%.4f'%(sum(lpips_score_mine[:(i+1)])/(i+1)))
    elif name=="VimeoValDataset":
        with torch.no_grad():
            lpips_score_mine, psnr_score_mine, msssim_score_mine, ssim_score_mine = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

            time_stamp = time.time()
            num = val_data.__len__()
            print("Dataset Length:", num)

            for i, data in tqdm(enumerate(val_data), desc="Processing", total=num):
                data_gpu, data_name = data
                data_gpu = data_gpu.to(device, non_blocking=True) / 255.
                preds = model.eval(data_gpu, name)

                b,n,c,h,w = preds.shape
                assert b==1 and n==1

                gt, pred = data_gpu[0], preds[0]
                if save_img:
                    pred_1_name = os.path.join(save_img_path, data_name[0])
                    print(pred_1_name, data_name)
                    if not os.path.exists(pred_1_name):
                        os.makedirs(pred_1_name)

                
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

                if save_img:
                    cv2.imwrite(os.path.join(pred_1_name, 'pred_%d.png'%(j+1)), pred_1[0])

            eval_time_interval = time.time() - time_stamp

            psnr_score_mine, ssim_score_mine, msssim_score_mine, lpips_score_mine = psnr_score_mine/num, ssim_score_mine/num, msssim_score_mine/num, lpips_score_mine/num
            for i in range(1):
                logger.info('%d             '%(nr_eval)+name+'  psnr_%d     '%(i)+'%.4f'%(sum(psnr_score_mine[:(i+1)])/(i+1))+'  ssim_%d     '%(i)+'%.4f'%(sum(ssim_score_mine[:(i+1)])/(i+1))+'  ms_ssim_%d     '%(i)+
                '%.4f'%(sum(msssim_score_mine[:(i+1)])/(i+1))+'  lpips_%d     '%(i)+'%.4f'%(sum(lpips_score_mine[:(i+1)])/(i+1)))
    
if __name__ == "__main__":    

    model = Model(load_path=args.load_path, training=False)
    test(model, args.save_img)
        
