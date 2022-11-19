# torch basis

1 torch.randn(784,10)

2 requires_grad参数：定义的时候就得写上

```python
bias = torch.zeros(10, requires_grad=True)
```

3 @表示矩阵相乘

```python
log_softmax(xb @ weights + bias)
```

4 数组作为index

```python
input[range(target.shape[0]), target].mean()
```

5 torch.argmax

```python
torch.argmax(out, dim=1)# 返回指定维度最大值的序号
```

6 tensor==tensor不是bool类型，是tensor类型

```python
(preds == yb).float().mean()
```

```python
print(type(pred1==yb))
print((pred1==yb))
>> <class 'torch.Tensor'>
>> tensor([False, False,  True, False, False, False, False, False, False, False])
```

7 设置断点

```python
IPython.core.debugger import set_trace
```

8 loss.backward()

```python
optimizer.zero_grad() # 清空过往梯度
loss.backward() # 反向传播，计算当前梯度
optimizer.step() # 根据梯度更新网络参数
```



```python
with torch.no_grad():
  # 表明当前计算不需要反向传播，使用之后，强制后边的内容不进行计算图的构建
  
```



```python
# tensor类型的数据(tensor_data):
a = tensor_data.grad # 返回tensor_data的梯度
tensor_data.grad.zero_() # 梯度清零，否则会积累
```



# torch.nn

## torch.nn.functional

简化代码的第一步：把激活函数和loss函数用torch.nn.functional替代:

```python
import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias
```

cross_entropy: 结合了负对数似然的loss函数和log softmax激活函数

验证可知，loss(model(xb), yb)和accuracy(model(xb), yb)都与之前手打的一样。

## nn.Module&nn.Parameter

## super().____init____()

```python
from torch import nn
class Mnist_Logistic(nn.Module):
  def __init__(self):
    super().__init__() # 调用nn.Module类中的init
    self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
    self.bias = nn.Parameter(torch.zeros(10))

  def forward(self, xb):
    return xb @ self.weights + self.bias
```

**super()**是用来调用父类(基类)的方法，__init__()是类的构造方法

**super().init()** 就是调用**父类的init**方法， 同样可以使用super()去调用父类的其他方法。

## fit()

打包我们的training loop得到fit函数，便于后续使用

```python
def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()
```

### forward()

自动实现

只要在实例化一个对象中传入对应参数，就可以自动调用forward函数。



## nn.Linear

Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

可以自己定义权重的均值和方差，但nn.linear内部是自己生成权重的，不能全部自己定义。

### Args:

​        **in_features**: size of each input sample

​        **out_features**: size of each output sample

​        **bias**: If set to ``False``, the layer will not learn an additive bias.

​           	  Default: ``True``



### class Linear(Module):



```python
def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

```

## torch.optim

contain various optimization algorithms

### Optimizer Ex:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer.zero_grad()
loss_fn(model(input), target).backward() optimizer.step()
```



```python
# 本来：
with torch.no_grad():
  for p in model.parameters():
    p -= p.grad*lr
  model.zero_grad()
# 现在：
opt.step()
opt.zero_grad()# 重新把梯度设为0
```



### opt.step()



### opt.zero_grad()

## Dataset

```python
from torch.utils.data import TensorDataset
train_ds = TensorDataset(x_train, y_train)
        
```



```python
# 原来：
xb = x_train[start_i:end_i]
yb = y_train[start_i:end_i]
# 现在xb,yb可以一起取了
xb, yb = train_ds[i*bs : i*bs+bs]
```



## DataLoader

```python
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
```



```python
# 原来的loop iterated over batches:
for i in range((n-1)//bs+1):
  xb,yb = train_ds[i*bs : i*bs+bs]
  pred = model(xb)
# 现在：
for epoch in range(epochs):
  #######这里开始#######
  for xb,yb in train_dl:
    pred = model(xb)
    loss = loss_func(pred, yb)
    
    loss.backward()
    opt.step()
    opt.zero_grad()
```

# Add validation

Need a Validation Set to identify if you are overfitting.

验证集

Shuffling the training data is important to prevent correlation between batches and overfitting. But validation set doesn't need shuffling.



