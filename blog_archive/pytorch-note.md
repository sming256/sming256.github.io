---
title: pytorch笔记
date: 2019-03-28 22:10:00
toc: true
category: Tech Zoo
tags:
---
本文记录了常用的pytorch命令，以供查询。

<!--more-->

* 网络结构中除batch维，进行维度平滑
```python
def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimensio
    nnum_features = 1
    for s in size:
        num_features *= s
    return num_features

x = x.view(-1, self.num_flat_features(x))
```
* torch.nn only supports mini-batches. If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.如果要把单个样本作为输入，需要增加维度

* 可以用loss.grad_fn输出计算图
```python
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d-> view -> linear -> relu -> linear -> relu -> linear-> MSELoss-> loss
```

* 在loss之后进行optimizer的迭代
```python
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update
```

* pytorch官方推荐先用numpy读入数据，再讲numpy转换为tensor


* 对图像的预处理工作大部分可以使用torch.transforms解决。数据增强在dataloader的transform里定义
```python
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomVerticalFlip(),
                       transforms.RandomRotation(15),
                       transforms.RandomRotation([90, 180, 270]),
                       transforms.Resize([32, 32]),
                       transforms.RandomCrop([28, 28]),
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
```

* 画图还是推荐使用plt
```python
import matplotlib.pyplot as plt
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()
```

* 一个标准的训练过程
```python
for epoch in range(2): # loop over the dataset multiple times
running_loss = 0.0
for i, data in enumerate(trainloader, 0):
# get the inputs
inputs, labels = data

# zero the parameter gradients
optimizer.zero_grad()

# forward + backward + optimize
outputs = net(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

# print statistics
running_loss += loss.item()
if i % 2000 == 1999: # print every 2000 mini-batches
print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))r
unning_loss = 0.0

print('Finished Training')
```

* 多GPU并行
```python
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)model.to(device)
```

* model save load
```python
# save
torch.save(model.state_dict(), PATH)

# load
model = TheModelClass(*args,**kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
# A common PyTorch convention is to save models using either a .pt or .pth file extension.

# PS：如果要把save GPU load 到CPU上
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
```

* finetune
```python
# 设置不需要更新权值的层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
# 先设置不需要更新的层，再设置需要更新的层
model_ft = models.alexnet(pretrained=use_pretrained)set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

# 查看网络中需要训练的参数
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
# 对需要训练的参数设置优化器
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
```

* 判断是否有脏数据.With pytorch 0.4 there is also torch.isnan:
```python
torch.isnan(x)
tensor([ 0, 0, 1], dtype=torch.uint8)
```

* `torch.cuda.device_count()`看可用gpu数目
* 查看pytorch版本`torch.__version__ `
* 查看是否支持cuda`print('gpu:', torch.cuda.is_available())`, 返回True/False
* `torch.norm`可以计算向量的范数，任意p范数包括正无穷都可以计算，正无穷是计算max，默认2范数
* 判断tensor的数据类型 `isinstance(a, torch.FloatTensor)` , `isinstance(a, torch.cuda.FloatTensor)`
* tensor的常用属性和方法
```python
a.shape, a.size() # 注意a.size(1)==a.shape[1]
a.dim() #返回a的维度
a.numel() #返回a的总数
```

* 初始化一个常量数组`torch.full([d1,d2], c)`，即创建一个d1xd2的张量，并全部填充c
* torch生成range
```python
torch.arrange(start, end, step) # end is not included
torch.linspace(start, end, steps=c) # end is included
torch.logspace(start, end, steps=c) # 对log而言均匀采样，实际上就是10^start到10^end
```

* 随机一个0到n-1的序列`torch.randperm(n) == random.shuffle(list(range(0,n)))`
* index也可以制定step，如`a[::2]`
* index_select可以在指定维度选择索引
```python
a = torch.rand(2,3,28,28)
a.index_select(2, torch.arrange(8)) 
# shape=[2,3,8,28]
# ...可以补充剩余的维度，不用再写多的
# 例如，a[0,...] a[...] a[:,1,...] a[...,:2]
```

* select by mask
```python
mask = a.ge(0.5) # same shape with a, bool
a[mask]  # selected shape
```
* select by flatten mask
```python
torch.take(a,torch.Tensor([0,2,5])
```
* expand和repeat
```python
# expand是指扩展到什么shape
a = torch.rand(1,32,5)
a.expand(4,32,10)  shape=[3,32,10]

# repeat是指每个维度复制几倍
a.repeate(4,32,10) shape=[4,32*32,10*5]
```
* 对于2D的tensor，a.t()表示转置
* 对于3D及以上，只转置两个维度，用a.transpose(dim0,dim1)，或者用permute(0,3,2,1) 任意组织维度顺序
* 但转置后有可能打乱存储顺序，要.contiguous()

* 比较两个tensor每个位置都相同，用torch.all(torch.eq(a,b))

* 两个tensor可以broadcast的判定：从最后一个维度往前数，要么维度对应的数值相同，要么有一个为1，要么不存在
```python
x=torch.empty(5,3,4,1)
y=torch.empty(  3,1,1) # x and y are broadcastable.
y=torch.empty(  2,1,1) # x and y are not broadcastable.
```
* torch中的乘法
  * 两个维度相同的矩阵的对应位置相乘（ 分素乘积 elementwiseproduct，Hadamard product）
    * a * b
    * torch.mul(a,b)
    * 上述两个a,b均可broadcast
  * 矩阵乘法（matrix multiplication）
    * torch.mm-只针对两个2d
    * torch.bmm-只针对两个3d实现batch mul（一定要注意bmm和mm的区别）
	* torch.matmu(a,b) == a@b
      * 最后两维是矩阵乘法，之前的维度broadcast后实现batch mul
      * 因此是两个3维之间乘法，等价于bmm
      * 对于1维和2维之间的乘法，之间查文档
    * torch.tensordot # TODO

* 数值计算
```python
a.floor() a.ceil()
a.round()四舍五入
a.trunc()整数 a.frac()小数
```
* 限制范围`grad.clamp(min,max=inf)`
* `a.prod()`返回a所有元素的乘积
* `torch.eq(a,b)` 等价于a==b，返回的是每个位置的0/1, 而`torch.equal(a,b)`是所有都为1则返回true，否则false
* topk 与 kthvalue
```python
value, index = a.topk(3,dim=1,largest=True) 前k大
value, index = a.kthvalue(8,dim=1)第k大
```
* torch.where(condition, x,y)
condition是一个0/1的和xy同shape的矩阵
如果为1，则选x，否则选y

* 设置模型的l2正则：
```python
optim.SGD(net.parameters(), lr=lr, weight_decay=0.001)
l1正则：
l1_loss = 0
for param in model.parameters():
   l1_loss += torch.sum(torch.abs(param))
loss = loss + 0.001 * regularization_loss
```

* 卷积网络输出
out =(( in+2*padding - dilation * (kernel-1))/stride).floor() + 1

* resnet 论文中指出，残差模块中，恒等映射之间至少要有两层网络，要不然效果不好
而且resnet可以降低网络参数，因为减少了Cin
两层resblock的写法
```
input ->conv1->bn->relu   -> conv2->bn-->sum->relu
          ->conv1x1->bn -----------------------|  #如果channel相同，直接相加
```

