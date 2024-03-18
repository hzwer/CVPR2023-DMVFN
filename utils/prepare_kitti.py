import os
import cv2
# train
kiki_train_path = '../data/KITTI/train_or'
kiki_train_path_new = '../data/KITTI/train'
dir_list = os.listdir(kiki_train_path)
dir_list.sort()
cnt = 0
for i in range(len(dir_list)):
    dir_path = os.path.join(kiki_train_path, dir_list[i])
    img_list = os.listdir(dir_path)
    img_list.sort()
    total_num = len(img_list)
    for i in range(total_num):
        if i<=3 or i>=(total_num-4):
            continue
        new_video_path = os.path.join(kiki_train_path_new, '%06d'%(cnt))
        os.makedirs(new_video_path)
        cnt += 1
        for j in range(i-4, i+5):
            im = cv2.imread(os.path.join(dir_path, img_list[j]))
            im_resized = cv2.resize(im, (832,256))
            cv2.imwrite(os.path.join(dir_path, img_list[j]), im_resized)
            os.system('cp %s %s'%(os.path.join(dir_path, img_list[j]), os.path.join(new_video_path, '%010d.png'%(j-i+4))))

kiki_train_path = '../data/KITTI/test_or'
kiki_train_path_new = '../data/KITTI/test'
dir_list = os.listdir(kiki_train_path)
dir_list.sort()
cnt = 0
for i in range(len(dir_list)):
    dir_path = os.path.join(kiki_train_path, dir_list[i])
    img_list = os.listdir(dir_path)
    img_list.sort()
    total_num = len(img_list)
    for i in range(total_num):
        if i<=3 or i>=(total_num-4):
            continue
        new_video_path = os.path.join(kiki_train_path_new, '%06d'%(cnt))
        os.makedirs(new_video_path)
        cnt += 1
        for j in range(i-4, i+5):
            im = cv2.imread(os.path.join(dir_path, img_list[j]))
            im_resized = cv2.resize(im, (832,256))
            cv2.imwrite(os.path.join(dir_path, img_list[j]), im_resized)
            os.system('cp %s %s'%(os.path.join(dir_path, img_list[j]), os.path.join(new_video_path, '%010d.png'%(j-i+4))))
