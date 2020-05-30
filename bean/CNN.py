import torch
from torch.autograd import Variable 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os 
from PIL import Image 
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
from time import sleep


#================================================================================================
# <CUDA 설정>
# 만약 GPU를 사용 가능하다면 device 값이 cuda가 되고, 아니라면 cpu가 됩니다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device : " + device)
print("="*50)
# 랜덤 시드 고정
torch.manual_seed(777)
# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
#================================================================================================


#================================================================================================
# <데이터 로드 (로컬)>


# X
normal_name_list = glob.glob('./Final_data/normal/*.png')
broken_name_list = glob.glob('./Final_data/broken/*.png')
black_name_list = glob.glob('./Final_data/black/*.png')

img_filename_list = np.array(normal_name_list + broken_name_list + black_name_list)
# img_filename = img_filename_list.reshape((img_filename_list.shape[0]),1)

# Y 
# 0 : normal
# 1 : broken
# 2 : black
normal_label = [ 0 for label in range(len(normal_name_list))]
broken_label = [ 1 for label in range(len(broken_name_list))]
black_label =  [ 2 for label in range(len(black_name_list))]

label_list = np.array(normal_label + broken_label + black_label)

# print(img_filename.shape, label.shape) # (63888, 1) (63888, 1)

CNT_DATA = len(label_list)
img_list = list()
for i in range(CNT_DATA):
    img = np.array(Image.open(img_filename_list[i]).convert('L'))
    img_list.append(img)
    # if i == 100: break


X_data = torch.Tensor(np.array(img_list)).unsqueeze(1)
# y_data = torch.LongTensor(label_list.reshape((label_list.shape[0],1)))
y_data = torch.from_numpy(label_list).long()
# print(img_list.size()) # (101, 64, 64, 1)

X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size = 0.25, random_state = 55)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    




#================================================================================================




#================================================================================================
dataset = TensorDataset(X_train, y_train)
type(dataset)
#================================================================================================


#================================================================================================
# <학습 환경 설정>
learning_rate = 0.001
training_epochs = 100
# batch_size = int(63888 / training_epochs)
batch_size = 100

data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
#================================================================================================


#================================================================================================
# <CNN 모델>
class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 64, 64, 1)
        #    Conv     -> (?, 64, 64, 32)
        #    Pool     -> (?, 32, 32, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1 ),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 32, 32, 32)
        #    Conv      ->(?, 32, 32, 64)
        #    Pool      ->(?, 16, 16, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 세번째층
        # ImgIn shape=(?, 16, 16, 64)
        #    Conv      ->(?, 16, 16, 128)
        #    Pool      ->(?, 8, 8, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 8x8x128 inputs -> 3 outputs
        self.fc = torch.nn.Linear(8 * 8 * 128, 3, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out
#================================================================================================

#================================================================================================
# <학습>

# CNN 모델 정의
model = CNN().to(device)
# 비용 함수와 옵티마이저를 정의합니다.
criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 총 배치의 수를 출력해보겠습니다.
total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))
# 총 배치의 수 : 600
# 총 배치의 수는 600입니다. 그런데 배치 크기를 100으로 했으므로 결국 훈련 데이터는 총 60,000개란 의미입니다. 

# 이제 모델을 훈련시켜보겠습니다.
model.train()

for epoch in range(training_epochs):
    
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        # image is already size of (64x64), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)

        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


# test dataset 만들기
testset = TensorDataset(X_test, y_test)
type(testset)

batch_size = 100

testloader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)

# 테스트 데이터로 모델 테스트 진행 
"""
https://wingnim.tistory.com/36
"""

model.eval()
test_loss = 0 
correct = 0

for data, target in testloader: 
    data = data.to(device)
    target = target.to(device) 

    output = model(data)

    # sum up batch loss 
    test_loss += criterion(output, target).data

    # get the index of the max log-probability 
    pred = output.data.max(1, keepdim = True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

test_loss /= len(testloader.dataset)/batch_size

print('\nTest set : Average loss : {: .4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))

# batch_size = CNT_DATA /100 (한 에포크에 100번 돌도록)
# (?, 64, 64, 1) > (?, 32, 32, 32) > (?, 16, 16, 64) > (?, 8, 8, 128) > 3
# first epoch
    # [Epoch:    1] cost = 0.848301888
    # [Epoch:    2] cost = 0.605040371
    # [Epoch:    3] cost = 0.514206707
    # [Epoch:    4] cost = 0.456617117
    # [Epoch:    5] cost = 0.405326366
    # [Epoch:    6] cost = 0.362584949
    # [Epoch:    7] cost = 0.325544119
    # [Epoch:    8] cost = 0.296771646
    # [Epoch:    9] cost = 0.275493056
    # [Epoch:   10] cost = 0.248582363
    # [Epoch:   11] cost = 0.243197829
    # [Epoch:   12] cost = 0.217068955
    # [Epoch:   13] cost = 0.197396234
    # [Epoch:   14] cost = 0.189239979
    # [Epoch:   15] cost = 0.174016967
    # [Epoch:   16] cost = 0.169131741
    # [Epoch:   17] cost = 0.158653781
    # [Epoch:   18] cost = 0.142494902
    # [Epoch:   19] cost = 0.147569418
    # [Epoch:   20] cost = 0.13186419
    # [Epoch:   21] cost = 0.122663364
    # [Epoch:   22] cost = 0.118097417
    # [Epoch:   23] cost = 0.107574008
    # [Epoch:   24] cost = 0.0982827172
    # [Epoch:   25] cost = 0.093011722
    # [Epoch:   26] cost = 0.0841054618
    # [Epoch:   27] cost = 0.0762038529
    # [Epoch:   28] cost = 0.0732196867
    # [Epoch:   29] cost = 0.0729682818
    # [Epoch:   30] cost = 0.059184283
    # [Epoch:   31] cost = 0.055857271
    # [Epoch:   32] cost = 0.0537995063
    # [Epoch:   33] cost = 0.0441465974
    # [Epoch:   34] cost = 0.0396845266
    # [Epoch:   35] cost = 0.0354180746
    # [Epoch:   36] cost = 0.0340700112
    # [Epoch:   37] cost = 0.938652098
    # [Epoch:   38] cost = 1.08520555
    # [Epoch:   39] cost = 0.25232619
    # [Epoch:   40] cost = 0.2175477
    # [Epoch:   41] cost = 0.178022727
    # [Epoch:   42] cost = 0.155207545
    # [Epoch:   43] cost = 0.123249777
    # [Epoch:   44] cost = 0.101992726
    # [Epoch:   45] cost = 0.0923956186
    # [Epoch:   46] cost = 0.08036042
    # [Epoch:   47] cost = 0.0756195039
    # [Epoch:   48] cost = 0.0689679757
    # [Epoch:   49] cost = 0.0636030063
    # [Epoch:   50] cost = 0.0595835894
    # [Epoch:   51] cost = 0.0535204485
    # [Epoch:   52] cost = 0.0508710109
    # [Epoch:   53] cost = 0.0469349362
    # [Epoch:   54] cost = 0.0447341874
    # [Epoch:   55] cost = 0.0426355563
    # [Epoch:   56] cost = 0.0363516957
    # [Epoch:   57] cost = 0.0355288833
    # [Epoch:   58] cost = 0.0324238986
    # [Epoch:   59] cost = 0.0298698358
    # [Epoch:   60] cost = 0.0282925442
    # [Epoch:   61] cost = 0.0267700385
    # [Epoch:   62] cost = 0.0250759553
    # [Epoch:   63] cost = 0.0232499279
    # [Epoch:   64] cost = 0.023443833
    # [Epoch:   65] cost = 0.0205974039
    # [Epoch:   66] cost = 0.0198816694
    # [Epoch:   67] cost = 0.0197203215
    # [Epoch:   68] cost = 0.0176297575
    # [Epoch:   69] cost = 0.0159329697
    # [Epoch:   70] cost = 0.0168990511
    # [Epoch:   71] cost = 0.0141831217
    # [Epoch:   72] cost = 0.0145085044
    # [Epoch:   73] cost = 0.012909187
    # [Epoch:   74] cost = 0.0123230591
    # [Epoch:   75] cost = 0.0115448106
    # [Epoch:   76] cost = 0.0110041285
    # [Epoch:   77] cost = 0.0100295246
    # [Epoch:   78] cost = 0.00979445875
    # [Epoch:   79] cost = 0.00909798872
    # [Epoch:   80] cost = 0.00876094028
    # [Epoch:   81] cost = 0.00813693367
    # [Epoch:   82] cost = 0.00805336423
    # [Epoch:   83] cost = 0.00761295063
    # [Epoch:   84] cost = 0.00717052585
    # [Epoch:   85] cost = 0.00671434915
    # [Epoch:   86] cost = 0.00656320387
    # [Epoch:   87] cost = 0.00594290765
    # [Epoch:   88] cost = 0.00555944443
    # [Epoch:   89] cost = 0.00563878752
    # [Epoch:   90] cost = 0.00549102155
    # [Epoch:   91] cost = 0.00523166358
    # [Epoch:   92] cost = 0.00492165145
    # [Epoch:   93] cost = 0.00450764131
    # [Epoch:   94] cost = 0.00427164044
    # [Epoch:   95] cost = 0.00394388614
    # [Epoch:   96] cost = 0.00417557778
    # [Epoch:   97] cost = 0.0036919252
    # [Epoch:   98] cost = 0.00338166114
    # [Epoch:   99] cost = 0.00325041008
    # [Epoch:  100] cost = 0.00308118644

    # Test set : Average loss :  0.3298, Accuracy: 14553/15972 (91%)

# fc망
# 8 > 512(추가) -> 8
# 망을 하나 늘렸는데 더 안좋아짐.
# second epoch
    # [Epoch:    1] cost = 1.31636882
    # [Epoch:    2] cost = 0.588298559
    # [Epoch:    3] cost = 0.515225768
    # [Epoch:    4] cost = 0.428120941
    # [Epoch:    5] cost = 0.375841737
    # [Epoch:    6] cost = 0.329054952
    # [Epoch:    7] cost = 0.32080555
    # [Epoch:    8] cost = 3.06470871
    # [Epoch:    9] cost = 0.477823883
    # [Epoch:   10] cost = 0.409532458
    # [Epoch:   11] cost = 0.372237891
    # [Epoch:   12] cost = 0.338420004
    # [Epoch:   13] cost = 0.314553857
    # [Epoch:   14] cost = 0.286037087
    # [Epoch:   15] cost = 0.264295936
    # [Epoch:   16] cost = 0.258271009
    # [Epoch:   17] cost = 0.233940393
    # [Epoch:   18] cost = 0.220832482
    # [Epoch:   19] cost = 0.211179778
    # [Epoch:   20] cost = 0.201635361
    # [Epoch:   21] cost = 0.185560361
    # [Epoch:   22] cost = 0.179988354
    # [Epoch:   23] cost = 0.150302842
    # [Epoch:   24] cost = 0.135686576
    # [Epoch:   25] cost = 0.134006217
    # [Epoch:   26] cost = 0.120843649
    # [Epoch:   27] cost = 0.105884045
    # [Epoch:   28] cost = 0.0905714333
    # [Epoch:   29] cost = 0.107085519
    # [Epoch:   30] cost = 0.0855436176
    # [Epoch:   31] cost = 0.0800128132
    # [Epoch:   32] cost = 0.0622450896
    # [Epoch:   33] cost = 0.0703614503
    # [Epoch:   34] cost = 0.0487493575
    # [Epoch:   35] cost = 10.8827553
    # [Epoch:   36] cost = 0.640758395
    # [Epoch:   37] cost = 0.497174472
    # [Epoch:   38] cost = 0.434467137
    # [Epoch:   39] cost = 0.399787039
    # [Epoch:   40] cost = 0.369882762
    # [Epoch:   41] cost = 0.34440878
    # [Epoch:   42] cost = 0.322727501
    # [Epoch:   43] cost = 0.30489856
    # [Epoch:   44] cost = 0.279425085
    # [Epoch:   45] cost = 0.265652388
    # [Epoch:   46] cost = 0.271019667
    # [Epoch:   47] cost = 0.239187285
    # [Epoch:   48] cost = 0.222379982
    # [Epoch:   49] cost = 0.207930535
    # [Epoch:   50] cost = 0.205472469
    # [Epoch:   51] cost = 0.205779657
    # [Epoch:   52] cost = 0.187652603
    # [Epoch:   53] cost = 0.17201288
    # [Epoch:   54] cost = 0.167478114
    # [Epoch:   55] cost = 0.173201635
    # [Epoch:   56] cost = 0.156143233
    # [Epoch:   57] cost = 0.135996908
    # [Epoch:   58] cost = 0.128680274
    # [Epoch:   59] cost = 0.127835289
    # [Epoch:   60] cost = 0.128817961
    # [Epoch:   61] cost = 0.1169741
    # [Epoch:   62] cost = 0.102122732
    # [Epoch:   63] cost = 0.10352613
    # [Epoch:   64] cost = 0.0866759345
    # [Epoch:   65] cost = 0.0735587925
    # [Epoch:   66] cost = 0.146894038
    # [Epoch:   67] cost = 0.0824613944
    # [Epoch:   68] cost = 0.0580721162
    # [Epoch:   69] cost = 0.0522441529
    # [Epoch:   70] cost = 0.0442031324
    # [Epoch:   71] cost = 0.0457249358
    # [Epoch:   72] cost = 0.058008194
    # [Epoch:   73] cost = 0.0293931477
    # [Epoch:   74] cost = 0.0227511507
    # [Epoch:   75] cost = 0.0168689881
    # [Epoch:   76] cost = 0.0173524264
    # [Epoch:   77] cost = 2.50728536
    # [Epoch:   78] cost = 1.72575045
    # [Epoch:   79] cost = 0.340339303
    # [Epoch:   80] cost = 0.288026243
    # [Epoch:   81] cost = 0.249662906
    # [Epoch:   82] cost = 0.222860649
    # [Epoch:   83] cost = 0.211319163
    # [Epoch:   84] cost = 0.188078031
    # [Epoch:   85] cost = 0.16688633
    # [Epoch:   86] cost = 0.15492861
    # [Epoch:   87] cost = 0.150138006
    # [Epoch:   88] cost = 0.133108303
    # [Epoch:   89] cost = 0.117618896
    # [Epoch:   90] cost = 0.10639362
    # [Epoch:   91] cost = 0.103200272
    # [Epoch:   92] cost = 0.100952275
    # [Epoch:   93] cost = 0.093326278
    # [Epoch:   94] cost = 0.067487292
    # [Epoch:   95] cost = 0.0635878295
    # [Epoch:   96] cost = 0.0783471167
    # [Epoch:   97] cost = 0.0528509393
    # [Epoch:   98] cost = 0.0447657593
    # [Epoch:   99] cost = 0.0460230708
    # [Epoch:  100] cost = 0.0432918072

    # Test set : Average loss :  0.4893, Accuracy: 14081/15972 (88%)

# batch size = 100(한 에포크 : 479번 반복)
# third epoch
    # [Epoch:    1] cost = 0.584881186
    # [Epoch:    2] cost = 0.383236676
    # [Epoch:    3] cost = 0.315137774
    # [Epoch:    4] cost = 0.257810801
    # [Epoch:    5] cost = 0.230015725
    # [Epoch:    6] cost = 0.194984853
    # [Epoch:    7] cost = 0.168412626
    # [Epoch:    8] cost = 0.147951454
    # [Epoch:    9] cost = 0.132406771
    # [Epoch:   10] cost = 0.110452443
    # [Epoch:   11] cost = 0.112850502
    # [Epoch:   12] cost = 0.0868574083
    # [Epoch:   13] cost = 0.0765848979
    # [Epoch:   14] cost = 0.0793330967
    # [Epoch:   15] cost = 0.0921200663
    # [Epoch:   16] cost = 0.0550482683
    # [Epoch:   17] cost = 0.079389222
    # [Epoch:   18] cost = 0.082281746
    # [Epoch:   19] cost = 0.073030673
    # [Epoch:   20] cost = 0.0620206594
    # [Epoch:   21] cost = 0.0647966489
    # [Epoch:   22] cost = 0.0461172536
    # [Epoch:   23] cost = 0.0810820982
    # [Epoch:   24] cost = 0.0686237738
    # [Epoch:   25] cost = 0.047171399
    # [Epoch:   26] cost = 0.0879836157
    # [Epoch:   27] cost = 0.0734742731
    # [Epoch:   28] cost = 0.0680242181
    # [Epoch:   29] cost = 0.0816127211
    # [Epoch:   30] cost = 0.0738635287
    # [Epoch:   31] cost = 0.0608886629
    # [Epoch:   32] cost = 0.0602017902
    # [Epoch:   33] cost = 0.0650190711
    # [Epoch:   34] cost = 0.057561703
    # [Epoch:   35] cost = 0.0885506272
    # [Epoch:   36] cost = 0.0720812306
    # [Epoch:   37] cost = 0.0723821297
    # [Epoch:   38] cost = 0.0664845631
    # [Epoch:   39] cost = 0.0665957332
    # [Epoch:   40] cost = 0.0732441321
    # [Epoch:   41] cost = 0.072572507
    # [Epoch:   42] cost = 0.0572425127
    # [Epoch:   43] cost = 0.0736704469
    # [Epoch:   44] cost = 0.0824613348
    # [Epoch:   45] cost = 0.0811390132
    # [Epoch:   46] cost = 0.0818268284
    # [Epoch:   47] cost = 0.0587818399
    # [Epoch:   48] cost = 0.0615740567
    # [Epoch:   49] cost = 0.053337153
    # [Epoch:   50] cost = 0.0845976248
    # [Epoch:   51] cost = 0.0807818025
    # [Epoch:   52] cost = 0.103103243
    # [Epoch:   53] cost = 0.0574079826
    # [Epoch:   54] cost = 0.0583685227
    # [Epoch:   55] cost = 0.0509151816
    # [Epoch:   56] cost = 0.0817526579
    # [Epoch:   57] cost = 0.0835259259
    # [Epoch:   58] cost = 0.0675361156
    # [Epoch:   59] cost = 0.102777347
    # [Epoch:   60] cost = 0.0617603436
    # [Epoch:   61] cost = 0.0715031475
    # [Epoch:   62] cost = 0.0764923915
    # [Epoch:   63] cost = 0.0768514052
    # [Epoch:   64] cost = 0.0829220936
    # [Epoch:   65] cost = 0.0636691004
    # [Epoch:   66] cost = 0.0726069584
    # [Epoch:   67] cost = 0.0816555694
    # [Epoch:   68] cost = 0.0830152258
    # [Epoch:   69] cost = 0.102492467
    # [Epoch:   70] cost = 0.0724087581
    # [Epoch:   71] cost = 0.0554393791
    # [Epoch:   72] cost = 0.0711486116
    # [Epoch:   73] cost = 0.0796137154
    # [Epoch:   74] cost = 0.0939578936
    # [Epoch:   75] cost = 0.0732198134
    # [Epoch:   76] cost = 0.0654492825
    # [Epoch:   77] cost = 0.0772060081
    # [Epoch:   78] cost = 0.0829508603
    # [Epoch:   79] cost = 0.0734640807
    # [Epoch:   80] cost = 0.0622041486
    # [Epoch:   81] cost = 0.0628766567
    # [Epoch:   82] cost = 0.107590877
    # [Epoch:   83] cost = 0.090397507
    # [Epoch:   84] cost = 0.0829390883
    # [Epoch:   85] cost = 0.0733409822
    # [Epoch:   86] cost = 0.0657575577
    # [Epoch:   87] cost = 0.0921202376
    # [Epoch:   88] cost = 0.0819296539
    # [Epoch:   89] cost = 0.0538294353
    # [Epoch:   90] cost = 0.0744252428
    # [Epoch:   91] cost = 0.0965181738
    # [Epoch:   92] cost = 0.0688148066
    # [Epoch:   93] cost = 0.0834988952
    # [Epoch:   94] cost = 0.0844230205
    # [Epoch:   95] cost = 0.0565650724
    # [Epoch:   96] cost = 0.0876718685
    # [Epoch:   97] cost = 0.0600860715
    # [Epoch:   98] cost = 0.0826754421
    # [Epoch:   99] cost = 0.0930308923
    # [Epoch:  100] cost = 0.0715328231

    # Test set : Average loss :  0.8559, Accuracy: 14097/15972 (88%)


# 학습속도가 빠르고 / 과적합의 우려가 있음
# 시도해볼만한 것
# learning_rate 조정
# Epoch 특정 지점에서 stop시키는 것
# batch_size 조정