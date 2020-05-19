import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pickle

"""#### GPU check"""

import os
import imageio

if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False
leave_log = True
if leave_log:
    result_dir = './output'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

"""### 1. 데이터 로드 & 전처리 방식 지정"""

#데이터 전처리 방식을 지정한다.
transform = transforms.Compose([
  transforms.ToTensor(), # 데이터를 파이토치의 Tensor 형식으로바꾼다.
  transforms.Normalize(mean=(0.5,), std=(0.5,)) # 픽셀값 0 ~ 1 -> -1 ~ 1
])

import glob
import os
import sys


dataPath = "./data"
after_imgs = glob.glob(os.path.join(dataPath+"/normal_processed_data",'*.jpg'))
# defect_imgs = glob.glob(os.path.join(dataPath+"/defect",'test_*'))
# imgs = after_imgs + defect_imgs
imgs = after_imgs
print(len(imgs))