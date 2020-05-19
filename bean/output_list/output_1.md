## data
- img_len = 1700(normal)
- batch_size = 50
- MinMaxScaler
- 30*30 random vertor

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
  nn.Dropout(0.5),
  nn.Linear(in_features=1024, out_features=512),
  nn.LeakyReLU(0.3, ),
  nn.Dropout(0.5),
  nn.Linear(in_features=512, out_features=256),
  nn.LeakyReLU(0.3, ),
  nn.Dropout(0.5),
  nn.Linear(in_features=256, out_features=1),
  nn.Sigmoid())
~~~
