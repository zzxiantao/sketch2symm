# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
from PIL import Image
import numpy as np
from skimage import feature
# parts = ['skin', 'hair', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'nose', 'u_lip', 'mouth', 'l_lip', 'neck', 
#             'cloth', 'hat', 'eye_g', 'ear_r', 'neck_l']
inner_parts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'nose', 'u_lip', 'mouth', 'l_lip', 'eye_g', 'hair']
root = 'imgs/train/imagenet'
sketch_folder = 'sketches'

def get_edges(edge, t):
    edge[:,1:] = edge[:,1:] | (t[:,1:] != t[:,:-1])
    edge[:,:-1] = edge[:,:-1] | (t[:,1:] != t[:,:-1])
    edge[1:,:] = edge[1:,:] | (t[1:,:] != t[:-1,:])
    edge[:-1,:] = edge[:-1,:] | (t[1:,:] != t[:-1,:])
    return edge

for i in range(30000):
    img = Image.open(os.path.join(root, 'images', str(i) + '.png')).resize((512, 512), resample=Image.BILINEAR)
    # 删除部位分割处理逻辑
    sketch_path = os.path.join(root, sketch_folder, str(i).zfill(5) + '.png')
    if os.path.exists(sketch_path):
        sketch = Image.open(sketch_path).convert('L')
        sketch_array = np.array(sketch)
        # 直接使用完整草图
        edges = (sketch_array > 128).astype(np.uint8) * 255  # 二值化处理
    else:
        edges = np.zeros(img.size, dtype=np.uint8)
    
    # 保存处理结果
    Image.fromarray(edges).save(os.path.join(root, sketch_folder, 'edges', str(i).zfill(5) + '.png'))