import os
import cv2
import ipdb
import numpy as np

ouput = 'final'
f_name_ls = ['max_50000', 'max_49000']
f_ls = os.listdir(f_name_ls[0])

for img_file in f_ls:
    img_path_ls = [os.path.join(f_name, img_file) for f_name in f_name_ls]
    imgs = [cv2.imread(img_path) for img_path in img_path_ls]
    
    img_sum = np.zeros(imgs[0].shape)
    for img in imgs:
        img_sum += img

    img_sum = np.uint8(img_sum / len(f_name_ls))
    cv2.imwrite(os.path.join(ouput, img_file), img_sum)