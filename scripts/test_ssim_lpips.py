import os
import cv2
import math
import time
import torch
import numpy as np
import random
import argparse
from pytorch_msssim import ssim, ms_ssim
import lpips
import logging
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, save_dir, phase, level=logging.INFO, screen=False, to_file=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if to_file:
        log_file = os.path.join(save_dir, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

exp = os.path.abspath('.').split('/')[-1]
device = torch.device('cuda')
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
log_path = '../logs/results_test/{}'.format(exp)


if not os.path.exists(log_path):
    os.makedirs(log_path)

setup_logger('base', log_path, 'train', level=logging.INFO, screen=True, to_file=True)
logger = logging.getLogger('base')

root_gt_davis = '../data/DAVIS'
root_gt_kitti = '../data/KITTI/test'
root_gt_city = '../data/cityscapes/test'
root_gt_vimeo = '../data/vimeo_interp_test/target'

root_pred_city = '../data/dmvfn_final_final/city'
root_pred_kitti = '../data/dmvfn_final_final/kitti'
root_pred_vimeo = '../data/dmvfn_final_final/vimeo_test'
root_pred_davis = '../data/dmvfn_final_final/davis'

lpips_score_mine = np.zeros(5)
ssim_score_mine = np.zeros(5)

# city
video_city_list = os.listdir(root_gt_city)
logger.info('\n dmvfn_final_final/city')
for i in video_city_list:
    video_gt, video_pred = os.path.join(root_gt_city, i), os.path.join(root_pred_city, i)
    gts = sorted(os.listdir(video_gt))
    for j in range(5):
        pred, gt = cv2.imread(os.path.join(video_pred, 'pred_%d.png'%(j+1))), cv2.imread(os.path.join(video_gt, gts[j+4]))
        pred, gt = torch.from_numpy(pred.copy()).permute(2, 0, 1), torch.from_numpy(gt.copy()).permute(2, 0, 1)
        pred, gt = pred.to(device, non_blocking=True)/ 255., gt.to(device, non_blocking=True) / 255.

        pred, gt = pred.unsqueeze(dim=0), gt.unsqueeze(dim=0)

        psnr = -10 * math.log10(torch.mean((gt - pred) * (gt - pred)).cpu().data)
        ssim_val = ssim( gt, pred, data_range=1.0, size_average=False) # return (N,)
        ms_ssim_val = ms_ssim( gt, pred, data_range=1.0, size_average=False ) #(N,)
        x, y = ((gt-0.5)*2.0).clone(), ((pred-0.5)*2.0).clone()
        lpips_val = loss_fn_alex(x, y)
        
        lpips_score_mine[j] += lpips_val
        ssim_score_mine[j] += ms_ssim_val
for j in range(5):
    logger.info(' ms_ssim %.4f'%(sum(ssim_score_mine[:j+1])/(j+1)/(len(video_city_list))) + ' lpips %.4f'%(sum(lpips_score_mine[:j+1])/(j+1)/(len(video_city_list))))

lpips_score_mine = np.zeros(5)
ssim_score_mine = np.zeros(5)
# kitti
video_kitti_list = os.listdir(root_gt_kitti)
logger.info('\n dmvfn_final_final/kitti')
for i in video_kitti_list:
    video_gt, video_pred = os.path.join(root_gt_kitti, i), os.path.join(root_pred_kitti, i)
    gts = sorted(os.listdir(video_gt))
    for j in range(5):
        pred, gt = cv2.imread(os.path.join(video_pred, 'pred_%d.png'%(j+1))), cv2.imread(os.path.join(video_gt, gts[j+4]))
        pred, gt = torch.from_numpy(pred.copy()).permute(2, 0, 1), torch.from_numpy(gt.copy()).permute(2, 0, 1)
        pred, gt = pred.to(device, non_blocking=True)/ 255., gt.to(device, non_blocking=True) / 255.

        pred, gt = pred.unsqueeze(dim=0), gt.unsqueeze(dim=0)

        psnr = -10 * math.log10(torch.mean((gt - pred) * (gt - pred)).cpu().data)
        ssim_val = ssim( gt, pred, data_range=1.0, size_average=False) # return (N,)
        ms_ssim_val = ms_ssim( gt, pred, data_range=1.0, size_average=False ) #(N,)
        x, y = ((gt-0.5)*2.0).clone(), ((pred-0.5)*2.0).clone()
        lpips_val = loss_fn_alex(x, y)
        lpips_score_mine[j] += lpips_val
        ssim_score_mine[j] += ms_ssim_val
for j in range(5):
    logger.info(' ms_ssim %.4f'%(sum(ssim_score_mine[:j+1])/(j+1)/(len(video_kitti_list))) + ' lpips %.4f'%(sum(lpips_score_mine[:j+1])/(j+1)/(len(video_kitti_list))))

lpips_score_mine = np.zeros(5)
ssim_score_mine = np.zeros(5)
# davis
video_davis_list = os.listdir(root_gt_davis)
logger.info('\n dmvfn_final_final/davis')
for i in video_davis_list:
    video_gt, video_pred = os.path.join(root_gt_davis, i), os.path.join(root_pred_davis, i)
    gts = sorted(os.listdir(video_gt))
    for j in range(5):
        pred, gt = cv2.imread(os.path.join(video_pred, 'pred_%d.png'%(j+1))), cv2.imread(os.path.join(video_gt, gts[j+4]))
        pred, gt = torch.from_numpy(pred.copy()).permute(2, 0, 1), torch.from_numpy(gt.copy()).permute(2, 0, 1)
        pred, gt = pred.to(device, non_blocking=True)/ 255., gt.to(device, non_blocking=True) / 255.

        pred, gt = pred.unsqueeze(dim=0), gt.unsqueeze(dim=0)

        psnr = -10 * math.log10(torch.mean((gt - pred) * (gt - pred)).cpu().data)
        ssim_val = ssim( gt, pred, data_range=1.0, size_average=False) # return (N,)
        ms_ssim_val = ms_ssim( gt, pred, data_range=1.0, size_average=False ) #(N,)
        x, y = ((gt-0.5)*2.0).clone(), ((pred-0.5)*2.0).clone()
        lpips_val = loss_fn_alex(x, y)
        lpips_score_mine[j] += lpips_val
        ssim_score_mine[j] += ms_ssim_val
for j in range(5):
    logger.info(' ms_ssim %.4f'%(sum(ssim_score_mine[:j+1])/(j+1)/(len(video_davis_list))) + ' lpips %.4f'%(sum(lpips_score_mine[:j+1])/(j+1)/(len(video_davis_list))))



video_name_list_1 = os.listdir(root_gt_vimeo)

video_name_list = []
for i in video_name_list_1:
    x = os.listdir(os.path.join(root_gt_vimeo,i))
    for j in x:
        video_name_list.append(os.path.join(i,j))
video_name_list.sort()
lpips_score_mine = np.zeros(1)
ssim_score_mine = np.zeros(1)
logger.info('\n dmvfn_final_final/vimeo')
for i in video_name_list:
    video_gt, video_pred = os.path.join(root_gt_vimeo, i), os.path.join(root_pred_vimeo, i)
    gts = sorted(os.listdir(video_gt))
    for j in range(1):
        pred, gt = cv2.imread(os.path.join(video_pred, 'pred_%d.png'%(j+1))), cv2.imread(os.path.join(video_gt, gts[j+2]))
        pred, gt = torch.from_numpy(pred.copy()).permute(2, 0, 1), torch.from_numpy(gt.copy()).permute(2, 0, 1)
        pred, gt = pred.to(device, non_blocking=True)/ 255., gt.to(device, non_blocking=True) / 255.

        pred, gt = pred.unsqueeze(dim=0), gt.unsqueeze(dim=0)

        psnr = -10 * math.log10(torch.mean((gt - pred) * (gt - pred)).cpu().data)
        
        ssim_val = ssim( gt, pred, data_range=1.0, size_average=False) # return (N,)
        ms_ssim_val = ms_ssim( gt, pred, data_range=1.0, size_average=False ) #(N,)
        x, y = ((gt-0.5)*2.0).clone(), ((pred-0.5)*2.0).clone()
        lpips_val = loss_fn_alex(x, y)
        lpips_score_mine[j] += lpips_val
        ssim_score_mine[j] += ms_ssim_val
for j in range(1):
    logger.info(' ms_ssim %.4f'%(sum(ssim_score_mine[:j+1])/(j+1)/(len(video_name_list))) + ' lpips %.4f'%(sum(lpips_score_mine[:j+1])/(j+1)/(len(video_name_list))))
