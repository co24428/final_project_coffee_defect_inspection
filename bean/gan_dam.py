import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import fft, fftpack
import scipy.stats as st
from math import pi as PI

def createNormalizedGaussian2d(x_start, x_end, y_start, y_end, std ):
  A = (1/(2*PI*std**2))
  x = np.linspace(x_start,x_end, x_end - x_start + 1)
  y = np.linspace(y_start,y_end, y_end - y_start + 1)
  X,Y = np.meshgrid(x,y)
  f = A * np.exp(   -(   (np.square(X) + np.square(Y))  /  (2*std**2)  )     )
  f_norm = f/f.max()
  return [X,Y,f,f_norm]
  


test = np.array(createNormalizedGaussian2d(-15, 14, -15, 14, 8))[3]
# test = np.array(list(map(lambda x: 1-x, test)))
test = np.ravel(test)


import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pickle

# 1700 = 50*34
# batch_size = 50 # 수가 딱 떨어져야 에러가 안난다!!!

# 3400 = 40*85
# 3400 = 50*68
batch_size = 40



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
normal_imgs = glob.glob(os.path.join(dataPath+"/normal_processed_data",'*.jpg'))
broken_imgs = glob.glob(os.path.join(dataPath+"/broken_processed_data",'*.jpg'))
imgs = normal_imgs + broken_imgs
# imgs = normal_imgs
normal_imgs_len = len(normal_imgs)
broken_imgs_len = len(broken_imgs)

"""
# 코드 정리하기!!!!!
"""

from PIL import Image
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

temp = Image.open(imgs[0])
temp2 = torch.LongTensor(np.array(temp))
temp2.shape

# y_train = np.array([0.]*26 + [1.]*26)
# y_train = np.array([0.]*img_num)
y_train = np.array([0.]*normal_imgs_len + [1.]*broken_imgs_len)
y_train = torch.LongTensor(y_train)

img_arr = list()
scaler = MinMaxScaler()

custom_guss = createNormalizedGaussian2d(-32,31 , -32, 31, 10)[3]

for img in imgs:
  temp = Image.open(img)
  temp = temp.convert("L")
  temp2 = np.array(temp)
  temp2 =  scaler.fit_transform(np.array(temp))
  img_arr.append(temp2)

# for img in imgs:
#   temp = Image.open(img)
#   temp = temp.convert("L")
#   temp2 = np.array(temp)
#   temp2 = (temp2 *custom_guss) / temp2.max()
#   img_arr.append(temp2)

a = torch.Tensor(np.array(img_arr)).cuda()

q1 = a.unsqueeze(1)

q1.size()

from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader

dataset = TensorDataset(q1, y_train)



"""# 코드 정리하기!!!

<hr>
"""
#데이터를 한번에 batch_size만큼만 가져오는 dataloader를 만든다.
dataloader =DataLoader(dataset, batch_size=batch_size, shuffle=True)

"""### 2. 모델 구축(생성자 & 구분자)"""

# 생성자는 랜덤 벡터 z를 입력으로 받아 가짜 이미지를 출력한다.
class Generator(nn.Module):

  # 네트워크 구조
    def __init__(self):
      super(Generator, self).__init__()
      self.main = nn.Sequential(
        nn.Linear(in_features=30*30, out_features=1024),
        nn.LeakyReLU(0.1),
        # nn.Dropout(0.1),
        nn.Linear(in_features=1024, out_features=2048),
        nn.LeakyReLU(0.1),
        # nn.Dropout(0.1),
        nn.Linear(in_features=2048, out_features=3072),
        nn.LeakyReLU(0.1),
        # nn.Dropout(0.1),
        nn.Linear(in_features=3072, out_features=64*64),
        nn.Tanh())
    
  # (batch_size x 100) 크기의 랜덤 벡터를 받아 
  # 이미지를 (batch_size x 1 x 28 x 28) 크기로 출력한다.
    def forward(self, inputs):
      return self.main(inputs).view(-1, 1, 64, 64)

"""#### 구분자(Discriminator) 구축
"""

# 구분자는 이미지를 입력으로 받아 이미지가 진짜인지 가짜인지 출력한다.
class Discriminator(nn.Module):
    
# 네트워크 구조
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
      nn.Linear(in_features=64*64, out_features=1024),
      nn.LeakyReLU(0.3, ),
      nn.Dropout(0.6),
      nn.Linear(in_features=1024, out_features=512),
      nn.LeakyReLU(0.3, ),
      nn.Dropout(0.6),
      nn.Linear(in_features=512, out_features=256),
      nn.LeakyReLU(0.3, ),
      nn.Dropout(0.6),
      nn.Linear(in_features=256, out_features=1),
      nn.Sigmoid())
    
  # (batch_size x 1 x 28 x 28) 크기의 이미지를 받아
  # 이미지가 진짜일 확률을 0~1 사이로 출력한다.
  def forward(self, inputs):
    inputs = inputs.view(-1, 64*64)
    return self.main(inputs)

# # 구분자는 이미지를 입력으로 받아 이미지가 진짜인지 가짜인지 출력한다.
# class Discriminator(nn.Module):
    
# # 네트워크 구조
#   def __init__(self):
#     super(Discriminator, self).__init__()
#     self.main = nn.Sequential(
#       nn.Linear(in_features=64*64, out_features=1024),
#       nn.LeakyReLU(0.1, ),
#       # nn.Dropout(0.1),
#       nn.Linear(in_features=1024, out_features=512),
#       nn.LeakyReLU(0.1, ),
#       # nn.Dropout(0.1),
#       nn.Linear(in_features=512, out_features=1),
#       nn.Sigmoid())
    
#   # (batch_size x 1 x 28 x 28) 크기의 이미지를 받아
#   # 이미지가 진짜일 확률을 0~1 사이로 출력한다.
#   def forward(self, inputs):
#     inputs = inputs.view(-1, 64*64)
#     return self.main(inputs)

"""#### 생성자 & 구분자 객체 생성"""

G = Generator()
D = Discriminator()

if use_gpu:
    print("=================gpu check=====================")
    print("using_GPU")
    print("=================gpu check=====================")
    G.cuda()
    D.cuda()

"""### 손실함수 & 최적화기법 지정
"""

# Binary Cross Entropy loss
# criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()

# 생성자의 매개 변수를 최적화하는 Adam optimizer
G_optimizer = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
# 구분자의 매개 변수를 최적화하는 Adam optimizer
D_optimizer = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

"""### 시각화 함수"""

# Commented out IPython magic to ensure Python compatibility.
# 학습 결과 시각화하기
# %matplotlib inline
from matplotlib import pyplot as plt
import numpy as np

def square_plot(data, path):

    if type(data) == list:
	    data = np.concatenate(data)
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = (((0, n ** 2 - data.shape[0]) ,
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data , padding, mode='constant' , constant_values=1)  # pad with ones (white)

    # tilethe filters into an image
    data = data.reshape((n , n) + data.shape[1:]).transpose((0 , 2 , 1 , 3) + tuple(range(4 , data.ndim + 1)))

    data = data.reshape((n * data.shape[1] , n * data.shape[3]) + data.shape[4:])

    plt.imsave(path, data, cmap='gray')

if leave_log:
    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    generated_images = []
    
z_fixed = Variable(torch.randn(5 * 5, 30*30), volatile=True)
if use_gpu:
    z_fixed = z_fixed.cuda()

"""### 모델 반복학습
"""

from random import *

# a = [0.] 
# b = [1.]
# # [uniform(-1.0, 1.0) ]
# z = a*103 + [uniform(-1.0, 1.0) ]*3 + \
#      a*26 + [uniform(-1.0, 1.0) ]*5 +  \
#      a*24 + [uniform(-1.0, 1.0) ]*7 +  \
#      a*22 + [uniform(-1.0, 1.0) ]*9 +  \
#      a*20 + [uniform(-1.0, 1.0) ]*11 +  \
#      a*17 + [uniform(-1.0, 1.0) ]*15 +  \
#      a*15 + [uniform(-1.0, 1.0) ]*15 + \
#      a*13+ [uniform(-1.0, 1.0) ]*19 + \
#      a*11 + [uniform(-1.0, 1.0) ]*19 + \
#      a*10 + [uniform(-1.0, 1.0) ]*21 + \
#      a*9 + [uniform(-1.0, 1.0) ]*21 + \
#      a*9 + [uniform(-1.0, 1.0) ]*21 + \
#      a*10 + [uniform(-1.0, 1.0) ]*21 + \
#      a*9 + [uniform(-1.0, 1.0) ]*21 + \
#      a*9 + [uniform(-1.0, 1.0) ]*21 + \
#      a*9 + [uniform(-1.0, 1.0) ]*20 + \
#      a*10 + [uniform(-1.0, 1.0) ]*19 + \
#      a*11 + [uniform(-1.0, 1.0) ]*18 + \
#      a*13 + [uniform(-1.0, 1.0) ]*16 + \
#      a*15 + [uniform(-1.0, 1.0) ]*13 + \
#      a*17 + [uniform(-1.0, 1.0) ]*12 + \
#      a*19 + [uniform(-1.0, 1.0) ]*10 + \
#      a*21 + [uniform(-1.0, 1.0) ]*8 + \
#      a*23 + [uniform(-1.0, 1.0) ]*6 + \
#      a*26 + [uniform(-1.0, 1.0) ]*3 + \
#      a*75
# z = torch.FloatTensor(z)
# z = z.unsqueeze(0)
# # batch_size
# z = torch.cat([z]*30, dim=0)

# if use_gpu:
#      z = z.cuda()

# a = np.array([1]* 900) * uniform(0,1)
# a = [1]*900
# z = list(map(lambda x: x*uniform(0,1), [1]*900))


# 데이터셋을 100번 돌며 학습한다.
for epoch in range(100):
    
    if leave_log:
        D_losses = []
        G_losses = []
    
    # 한번에 batch_size만큼 데이터를 가져온다.
    for real_data, _ in dataloader:
        batch_size = real_data.size(0)
        # print(batch_size)
        
        # 데이터를 pytorch의 변수로 변환한다.
        real_data = Variable(real_data)
        # print(real_data.size())

        ### 구분자 학습시키기

        # 이미지가 진짜일 때 정답 값은 1이고 가짜일 때는 0이다.
        # 정답지에 해당하는 변수를 만든다.
        target_real = Variable(torch.ones(batch_size, 1))
        target_fake = Variable(torch.zeros(batch_size, 1))
         
        if use_gpu:
            real_data, target_real, target_fake = real_data.cuda(), target_real.cuda(), target_fake.cuda()
            
        # 진짜 이미지를 구분자에 넣는다.
        D_result_from_real = D(real_data)
        # 구분자의 출력값이 정답지인 1에서 멀수록 loss가 높아진다.
        D_loss_real = criterion(D_result_from_real, target_real)

        # 생성자에 입력으로 줄 랜덤 벡터 z를 만든다.
        # z = Variable(torch.randn((batch_size, 900)))
        z = np.array(list(map(lambda x: x*uniform(0,1), [1]*900)))
        z = z * test
        z = torch.FloatTensor(z)
        z = z.unsqueeze(0)
        z = torch.cat([z]*batch_size, dim=0)

        if use_gpu:
            z = z.cuda()
            
        # 생성자로 가짜 이미지를 생성한다.
        fake_data = G(z)
        
        # 생성자가 만든 가짜 이미지를 구분자에 넣는다.
        D_result_from_fake = D(fake_data)
        # 구분자의 출력값이 정답지인 0에서 멀수록 lo`ss가 높아진다.
        D_loss_fake = criterion(D_result_from_fake, target_fake)
        
        # 구분자의 loss는 두 문제에서 계산된 loss의 합이다.
        D_loss = D_loss_real + D_loss_fake
        
        # 구분자의 매개 변수의 미분값을 0으로 초기화한다.
        D.zero_grad()
        # 역전파를 통해 매개 변수의 loss에 대한 미분값을 계산한다.
        D_loss.backward()
        # 최적화 기법을 이용해 구분자의 매개 변수를 업데이트한다.
        D_optimizer.step()
        
        if leave_log:
            # D_losses.append(D_loss.data[0])
            D_losses.append(D_loss.item())

        # train generator G

        ### 생성자 학습시키기
        
        # 생성자에 입력으로 줄 랜덤 벡터 z를 만든다.
        # z = Variable(torch.randn((batch_size, 900)))
        z = np.array(list(map(lambda x: x*uniform(0,1), [1]*900)))
        z = z * test
        z = torch.FloatTensor(z)
        z = z.unsqueeze(0)
        z = torch.cat([z]*batch_size, dim=0)
        
        if use_gpu:
            z = z.cuda()
        
        # 생성자로 가짜 이미지를 생성한다.
        fake_data = G(z)
        # 생성자가 만든 가짜 이미지를 구분자에 넣는다.
        D_result_from_fake = D(fake_data)
        # 생성자의 입장에서 구분자의 출력값이 1에서 멀수록 loss가 높아진다.
        G_loss = criterion(D_result_from_fake, target_real)
        
        # 생성자의 매개 변수의 미분값을 0으로 초기화한다.
        G.zero_grad()
        # 역전파를 통해 매개 변수의 loss에 대한 미분값을 계산한다.
        G_loss.backward()
        # 최적화 기법을 이용해 생성자의 매개 변수를 업데이트한다.
        G_optimizer.step()
        
        if leave_log:
            # G_losses.append(G_loss.data[0])
            G_losses.append(G_loss.item())
    if leave_log:
        # true_positive_rate = (D_result_from_real > 0.5).float().mean().data[0]
        # true_negative_rate = (D_result_from_fake < 0.5).float().mean().data[0]
        true_positive_rate = (D_result_from_real > 0.5).float().mean().item()
        true_negative_rate = (D_result_from_fake < 0.5).float().mean().item()
        
        base_message = ("Epoch: {epoch:<3d} D Loss: {d_loss:<8.6} G Loss: {g_loss:<8.6} "
                        "True Positive Rate: {tpr:<5.1%} True Negative Rate: {tnr:<5.1%}"
                       )
        message = base_message.format(
                    epoch=epoch,
                    d_loss=sum(D_losses)/len(D_losses),
                    g_loss=sum(G_losses)/len(G_losses),
                    tpr=true_positive_rate,
                    tnr=true_negative_rate
        )
        print(message)
    
    if leave_log:
        fake_data_fixed = G(z_fixed)
        image_path = result_dir + '/epoch{}.png'.format(epoch)
        square_plot(fake_data_fixed.view(25, 64, 64).cpu().data.numpy(), path=image_path)
        generated_images.append(image_path)
    
    if leave_log:
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

torch.save(G.state_dict(), "gan_generator.pkl")
torch.save(D.state_dict(), "gan_discriminator.pkl")
with open('gan_train_history.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

generated_image_array = [imageio.imread(generated_image) for generated_image in generated_images]
imageio.mimsave(result_dir + '/GAN_generation.gif', generated_image_array, fps=5)