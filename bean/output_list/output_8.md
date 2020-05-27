## data
- img_len = 11850(normal rotated binary)
- batch_size = 30
- epoch = 100

## image preprocessing ( Scaler )
~~~
temp = ((cv2.imread(img,cv2.IMREAD_GRAYSCALE)/255) *2) -1 # pixel : -1 ~ 1 
img_arr.append(temp)
~~~

## G network

~~~
super(Generator, self).__init__()
self.main = nn.Sequential(
  nn.Linear(in_features=64*64, out_features=2048),
  nn.LeakyReLU(0.1),
  # nn.Dropout(0.1),
  nn.Linear(in_features=2048, out_features=1024),
  nn.LeakyReLU(0.1),
  # nn.Dropout(0.1),
  nn.Linear(in_features=1024, out_features=2048),
  nn.LeakyReLU(0.1),
  # nn.Dropout(0.1),
  nn.Linear(in_features=2048, out_features=64*64),
  nn.Tanh())
~~~

## D network

~~~
self.main = nn.Sequential(
  nn.Linear(in_features=64*64, out_features=1024),
  nn.LeakyReLU(0.4, ),
  nn.Dropout(0.6),
  nn.Linear(in_features=1024, out_features=512),
  nn.LeakyReLU(0.4, ),
  nn.Dropout(0.6),
  nn.Linear(in_features=512, out_features=256),
  nn.LeakyReLU(0.4, ),
  nn.Dropout(0.6),
  nn.Linear(in_features=256, out_features=1),
  nn.Sigmoid())
~~~

## D loss custom
~~~
D_loss = (D_loss_real + D_loss_fake) * 0.4
~~~

## vector custom
- random vector > 64*64

~~~
z = Variable(torch.randn((batch_size, 64*64)))
~~~


