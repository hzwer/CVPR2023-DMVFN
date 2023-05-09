import os
import cv2
# city test
city_test_root = '../data/cityscapes/leftImg8bit_sequence/val'
city_test_dir = os.listdir(city_test_root)
city_test_dir.sort()
city_test = '../data/cityscapes/test'
num = 0
for i in range(len(city_test_dir)):
    frame_dir = os.path.join(city_test_root, city_test_dir[i])
    frame_list = os.listdir(frame_dir)
    frame_list.sort()
    for j in range(len(frame_list)//30):
        print(num, len(frame_list))
        city_test_video_path = os.path.join(city_test, '%06d'%(num))
        if not os.path.exists(city_test_video_path):
            os.makedirs(city_test_video_path)
        for k in range(j*30, (j+1)*30):
            full_image_path = os.path.join(frame_dir, frame_list[k])
            assert os.path.isfile(full_image_path)
            im = cv2.imread(full_image_path)
            im_resized = cv2.resize(im, (1024,512))
            cv2.imwrite(full_image_path, im_resized)
            img_path = os.path.join(city_test_video_path, '%010d.png'%(k-(j*30)))
            os.system('cp %s %s'%(full_image_path, img_path))
        num += 1


# city train
city_train_root = '../data/cityscapes/leftImg8bit_sequence/train'
city_train_dir = os.listdir(city_train_root)
city_train_dir.sort()
city_train = '../data/cityscapes/train'
num = 0
for i in range(len(city_train_dir)):
    frame_dir = os.path.join(city_train_root, city_train_dir[i])
    frame_list = os.listdir(frame_dir)
    frame_list.sort()
    for j in range(len(frame_list)//30):
        city_train_video_path = os.path.join(city_train, '%06d'%(num))
        if not os.path.exists(city_train_video_path):
            os.makedirs(city_train_video_path)
        for k in range(j*30, (j+1)*30):
            full_image_path = os.path.join(frame_dir, frame_list[k])
            assert os.path.isfile(full_image_path)
            im = cv2.imread(full_image_path)
            im_resized = cv2.resize(im, (1024,512))
            cv2.imwrite(full_image_path, im_resized)
            img_path = os.path.join(city_train_video_path, '%010d.png'%(k-(j*30)))
            os.system('cp %s %s'%(full_image_path, img_path))
        num += 1
