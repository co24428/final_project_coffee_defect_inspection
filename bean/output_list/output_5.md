## data
- img_len = 3400(normal\_sec,1700 + broken\_sec,1700)
- batch_size = 40
- MinMaxScaler
- epoch = 200

## G network

~~~
self.main = nn.Sequential(
  nn.Linear(in_features=30*30, out_features=1024),
  nn.LeakyReLU(0.2),
  # nn.Dropout(0.1),
  nn.Linear(in_features=1024, out_features=2048),
  nn.LeakyReLU(0.2),
  # nn.Dropout(0.1),
  nn.Linear(in_features=2048, out_features=3072),
  nn.LeakyReLU(0.2),
  # nn.Dropout(0.1),
  nn.Linear(in_features=3072, out_features=64*64),
  nn.Tanh())
~~~

## D network

~~~
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
~~~

## vector custom

~~~
def createNormalizedGaussian2d(x_start, x_end, y_start, y_end, std ):
  A = (1/(2*PI*std**2))
  x = np.linspace(x_start,x_end, x_end - x_start + 1)
  y = np.linspace(y_start,y_end, y_end - y_start + 1)
  X,Y = np.meshgrid(x,y)
  f = A * np.exp(   -(   (np.square(X) + np.square(Y))  /  (2*std**2)  )     )
  f_norm = f/f.max()
  return [X,Y,f,f_norm]
  
test = np.array(createNormalizedGaussian2d(-15, 14, -15, 14, 8))[3]
test = np.ravel(test)

...

z = np.array(list(map(lambda x: x*uniform(0,1), [1]*900)))
z = z * test
z = torch.FloatTensor(z)
z = z.unsqueeze(0)
z = torch.cat([z]*batch_size, dim=0)


~~~


