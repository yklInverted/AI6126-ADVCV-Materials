import os
import cv2

f_name = 'max_46000'
f_ls = os.listdir(f_name)

for img_file in f_ls:
    img_path = os.path.join(f_name, img_file)
    img = cv2.imread(img_path)
    
    cv2.imwrite(img_path, img[:,-512:,:])
    print(os.path.join(f_name, img_file[:5] + '.png'))
    os.rename(img_path, os.path.join(f_name, img_file[:5] + '.png'))