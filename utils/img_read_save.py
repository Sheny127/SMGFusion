import numpy as np
import cv2
import os
from skimage.io import imsave

def image_read_cv2(path, mode='RGB'):
    # 保持原有的读取逻辑，读取出来的数据范围是 0-255 (float32)
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image, imagename, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    image = np.squeeze(image)
    
    # --- 修复逻辑：自动判断范围 ---
    if image.dtype == np.float32 or image.dtype == np.float64:
        # 如果数据中有大于 1 的值，说明它是 0-255 范围
        if image.max() > 1.1: 
            image = np.clip(image, 0, 255).astype(np.uint8)
        # 否则，说明它是 0-1 范围，需要乘 255
        else:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
    # ---------------------------

    imsave(os.path.join(savepath, "{}.png".format(imagename)), image)