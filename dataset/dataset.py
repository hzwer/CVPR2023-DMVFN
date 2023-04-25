import cv2
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CityTrainDataset(Dataset):
    def __init__(self):
        self.path = './data/cityscapes/train'
        self.train_data = sorted(os.listdir(self.path))


    def __len__(self):
        return len(self.train_data)

    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs

    def getimg(self, index):
        data_name = self.train_data[index]
        data_path = os.path.join(self.path, data_name)
        frame_list = sorted(os.listdir(data_path))
        imgs = []
        for i in range(30):
            im = cv2.imread(os.path.join(data_path, frame_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        imgs = self.aug_seq(imgs, 256, 256)
        length = len(imgs)
        if random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, ::-1]
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)
        return torch.stack(imgs, 0)#n, c, h , w

class CityValDataset(Dataset):
    def __init__(self):
        self.val_data = []
        self.video_path = './data/cityscapes/test'
        self.video_data = sorted(os.listdir(self.video_path))
        for i in self.video_data:
            self.val_data.append(os.path.join(self.video_path, i))
        self.val_data = sorted(self.val_data)

    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data))
        imgs = []
        for i in range(14):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        name = self.video_data[index]
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0), name#n, c, h , w

class KittiTrainDataset(Dataset):
    def __init__(self):
        self.path = './data/KITTI/train'
        self.train_data = sorted(os.listdir(self.path))

    def __len__(self):
        return len(self.train_data)


    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs

    def getimg(self, index):
        data_name = self.train_data[index]
        data_path = os.path.join(self.path, data_name)
        frame_list = sorted(os.listdir(data_path))
        imgs = []
        for i in range(9):
            im = cv2.imread(os.path.join(data_path, frame_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        imgs = self.aug_seq(imgs, 256, 256)
        length = len(imgs)
        if random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, ::-1]
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0)#n, c, h , w

class KittiValDataset(Dataset):
    def __init__(self):
        self.val_data = []
        self.video_path = './data/KITTI/test'
        self.video_data = sorted(os.listdir(self.video_path))
        for i in self.video_data:
            self.val_data.append(os.path.join(self.video_path, i))
        self.val_data = sorted(self.val_data)

    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data))
        imgs = []
        for i in range(9):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        name = self.video_data[index]
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0), name#n, c, h , w

class VimeoTrainDataset(Dataset):
    def __init__(self):
        self.path = './data/Vimeo/sequences/'
        self.train_data = []
        video_paths = sorted(os.listdir(self.path))
        for i in video_paths:
            video_path_1 = sorted(os.listdir(os.path.join(self.path, i)))
            for j in video_path_1:
                self.train_data.append(os.path.join(os.path.join(self.path, i), j))
        

    def __len__(self):
        return len(self.train_data)         

    def aug(self, img0, img1, gt, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, img1, gt

    def getimg(self, index):
        data = self.train_data[index]
        frame_list = sorted(os.listdir(data))
        ind = [0, 1, 2, 3, 4]
        random.shuffle(ind)
        ind = ind[0]
        img0 = cv2.imdread(frame_list[ind])
        img1 = cv2.imdread(frame_list[ind+1])
        gt = cv2.imdread(frame_list[ind+2])
        return img0, img1, gt
            
    def __getitem__(self, index):
        img0, img1, gt = self.getimg(index)
        img0, img1, gt = self.aug(img0, img1, gt, 224, 224)
        if random.randint(0, 1):
            img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
            gt = cv2.rotate(gt, cv2.ROTATE_180)
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
        elif random.randint(0, 1):
            img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, :, ::-1]
            img1 = img1[:, :, ::-1]
            gt = gt[:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            img0 = img0[::-1]
            img1 = img1[::-1]
            gt = gt[::-1]
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, ::-1]
            img1 = img1[:, ::-1]
            gt = gt[:, ::-1]
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.stack((img0, img1, gt), 0)


class UCFTrainDataset(Dataset):
    def __init__(self):
        self.path = './data/ucf101_jpeg/jpegs_256/'
        self.train_data = []
        with open(os.path.join('/home/huxiaotao/trainlist01.txt')) as f:
            for line in f:
                video_dir = line.rstrip().split('.')[0]
                video_name = video_dir.split('/')[1]
                self.train_data.append(video_name)

    def __len__(self):
        return len(self.train_data)

    def aug_seq(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs

    def getimg(self, index):
        video_path = self.train_data[index]
        frame_list = sorted(os.listdir(os.path.join(self.path, video_path)))
        n = len(frame_list)
        max_time = 5 if 5 <= n/10 else int(n/10)
        time_step = np.random.randint(1, max_time + 1)#1, 2, 3, 4, 5
        frame_ind = np.random.randint(0, n-9*time_step)
        frame_inds = [frame_ind+j*time_step for j in range(10)]
        imgs = []
        for i in frame_inds:
            im = cv2.imread(os.path.join(os.path.join(self.path, video_path), frame_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        imgs = self.aug_seq(imgs, 256, 256)
        length = len(imgs)
        if random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
        elif random.randint(0, 1):
            for i in range(length):
                imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][::-1]
        if random.uniform(0, 1) < 0.5:
            for i in range(length):
                imgs[i] = imgs[i][:, ::-1]
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0)#n, c, h , w

class VimeoValDataset(Dataset):
    def __init__(self):
        self.val_data = []
        self.video_path = './data/vimeo_interp_test/target/'
        self.data_name = []
        self.video_list = sorted(os.listdir(self.video_path))
        for i in self.video_list:
            self.video_clip_list = sorted(os.listdir(os.path.join(self.video_path, i)))
            for j in self.video_clip_list:
                self.val_data.append(os.path.join(self.video_path, os.path.join(i, j)))
                self.data_name.append(os.path.join(i, j))
        self.val_data = sorted(self.val_data)
        self.data_name = sorted(self.data_name)

    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data))
        imgs = []
        for i in range(3):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        data_name = self.data_name[index]
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)
        return torch.stack(imgs, 0), data_name#n, c, h , w

class DavisValDataset(Dataset):
    def __init__(self):
        self.val_data = []
        self.video_path = './data/DAVIS/'
        self.video_data = sorted(os.listdir(self.video_path))
        for i in self.video_data:
            self.val_data.append(os.path.join(self.video_path, i))
        self.val_data = sorted(self.val_data)

    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data)) #一定要sort
        imgs = []
        for i in range(9):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        name = self.video_data[index]
        length = len(imgs)
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)

        return torch.stack(imgs, 0), name#n, c, h , w