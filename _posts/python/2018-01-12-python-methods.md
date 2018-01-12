---
layout: post
title: Python方法收集
category: PYTHON
tags: python
description: 用来收集python常用的方法
---

1. 指定行和列，这里0是列
#>>> b = np.arange(12).reshape(3,4)
#>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> b.sum(axis=0)                            # sum of each column
array([12, 15, 18, 21])
>>>
>>> b.min(axis=1)                            # min of each row
array([0, 4, 8])
>>>
>>> b.cumsum(axis=1)                         # cumulative sum along each row
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])

2. 返回相关索引，有重复的返回第一个
>>> a = np.arange(6).reshape(2,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.argmax(a)
5
>>> np.argmax(a, axis=0)
array([1, 1, 1])
>>> np.argmax(a, axis=1)
array([2, 2])
>>>
>>> b = np.arange(6)
>>> b[1] = 5
>>> b
array([0, 5, 2, 3, 4, 5])
>>> np.argmax(b) # Only the first occurrence is returned.
1

3.传入方法根据方法进行操作
>>> def my_func(a):
...     """Average first and last element of a 1-D array"""
...     return (a[0] + a[-1]) * 0.5
>>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])
>>> np.apply_along_axis(my_func, 0, b)
array([ 4.,  5.,  6.])
>>> np.apply_along_axis(my_func, 1, b)
array([ 2.,  5.,  8.])

>>> b = np.array([[8,1,7], [4,3,9], [5,2,6]])
>>> np.apply_along_axis(sorted, 1, b)
array([[1, 7, 8],
       [3, 4, 9],
       [2, 5, 6]])

4. 传入方法操作 这里1是列 0是行
>>> def f(x,y):
...     return 10*x+y
...
>>> b = np.fromfunction(f,(5,4),dtype=int)
>>> b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
>>> b[2,3]
23
>>> b[0:5, 1]                       # each row in the second column of b
array([ 1, 11, 21, 31, 41])
>>> b[ : ,1]                        # equivalent to the previous example
array([ 1, 11, 21, 31, 41])
>>> b[1:3, : ]                      # each column in the second and third row of b
array([[10, 11, 12, 13],
       [20, 21, 22, 23]])

>>> b[-1]                                  # the last row. Equivalent to b[-1,:]
array([40, 41, 42, 43])

terating over multidimensional arrays is done with respect to the first axis:

>>>
>>> for row in b:
...     print(row)
...
[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]
However, if one wants to perform an operation on each element in the array, one can use the flat attribute which is an iterator over all the elements of the array:

>>>
>>> for element in b.flat:
...     print(element)

>>> y = np.arange(35).reshape(5,7)
y = np.arange(35).reshape(5,7)

y
Out[47]: 
array([[ 0,  1,  2, ...,  4,  5,  6],
       [ 7,  8,  9, ..., 11, 12, 13],
       [14, 15, 16, ..., 18, 19, 20],
       [21, 22, 23, ..., 25, 26, 27],
       [28, 29, 30, ..., 32, 33, 34]])
>>> y[1:5:2,::3]
array([[ 7, 10, 13],
       [21, 24, 27]])

5. 矩阵变化
   >>> a = np.floor(10*np.random.random((3,4)))
>>> a
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
>>> a.shape
(3, 4)

>>> a.ravel()  # returns the array, flattened
array([ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.])
>>> a.reshape(6,2)  # returns the array with a modified shape

>>> a.resize((2,6))
>>> a
array([[ 2.,  8.,  0.,  6.,  4.,  5.],
       [ 1.,  1.,  8.,  9.,  3.,  6.]])

If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated:

>>>
>>> a.reshape(3,-1)
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
6.水平竖直拼接
>>> a = np.floor(10*np.random.random((2,2)))
>>> a
array([[ 8.,  8.],
       [ 0.,  0.]])
>>> b = np.floor(10*np.random.random((2,2)))
>>> b
array([[ 1.,  8.],
       [ 0.,  4.]])
>>> np.vstack((a,b))
array([[ 8.,  8.],
       [ 0.,  0.],
       [ 1.,  8.],
       [ 0.,  4.]])
>>> np.hstack((a,b))
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])
>>> np.column_stack((a,b))   # With 2D arrays 水平拼接
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])

>>> a = np.array([4.,2.])
>>> b = np.array([2.,8.])
>>> a[:,newaxis]  # This allows to have a 2D columns vector
array([[ 4.],
       [ 2.]]) 打散


7. 构造 0 1矩阵，重构0 1矩阵
https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.zeros_like.html#numpy.zeros_like
>>> np.zeros(5)
array([ 0.,  0.,  0.,  0.,  0.])
>>> s = (2,2)
>>> np.zeros(s)
array([[ 0.,  0.],
       [ 0.,  0.]])

>>> x = np.arange(6)
>>> x = x.reshape((2, 3))
>>> x
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.zeros_like(x)
array([[0, 0, 0],
       [0, 0, 0]])


8. 等距离分割
logspace
>>> np.linspace(2.0, 3.0, num=5)
array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
>>> np.linspace(2.0, 3.0, num=5, endpoint=False)
array([ 2. ,  2.2,  2.4,  2.6,  2.8])
>>> np.linspace(2.0, 3.0, num=5, retstep=True)
(array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

9.从数组取值
>>> np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
array([[ True, False, False],
       [False,  True, False],
       [False, False,  True]], dtype=bool)
>>>
>>> np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])

10 根据索引取值，索引出某部分内容，返回值是索引值
x = np.arange(20).reshape(5, 4)

x
Out[14]: 
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19]])

row, col = np.indices((2, 3))



row
Out[17]: 
array([[0, 0, 0],
       [1, 1, 1]])

col
Out[18]: 
array([[0, 1, 2],
       [0, 1, 2]])

x[row, col]
Out[19]: 
array([[0, 1, 2],
       [4, 5, 6]])

np.mgrid[-1:1:5j]
array([-1. , -0.5,  0. ,  0.5,  1. ])

11 统计每个数出现的次数
>>> np.bincount(np.arange(5))
array([1, 1, 1, 1, 1])
>>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
array([1, 3, 1, 1, 0, 0, 0, 1])

x = np.array([0, 1, 1, 3, 2, 1, 7, 23])

x.size
Out[32]: 8

np.bincount(x).size
Out[33]: 24

np.amax(x)+1
Out[34]: 24


>>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
>>> x = np.array([0, 1, 1, 2, 2, 2])
>>> np.bincount(x,  weights=w)
array([ 0.3,  0.7,  1.1])
根据权重所在数每个家0.7=0.5+0.2  1.1=0.7+1-0.6
out[n] += weight[i] instead of out[n] += 1.

12  取接近的数
>>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
>>> np.ceil(a)
array([-1., -1., -0.,  1.,  2.,  2.,  2.])  取大

>>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
>>> np.floor(a)
array([-2., -2., -1.,  0.,  1.,  1.,  2.])取小

>>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
>>> np.trunc(a)
array([-1., -1., -0.,  0.,  1.,  1.,  2.]) 截掉小数

>>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
>>> np.rint(a)
array([-2., -2., -0.,  0.,  2.,  2.,  2.]) 四舍五入


















数据处理
1. 修改nan 和缺省值
http://machinelearningmastery.com/handle-missing-timesteps-sequence-prediction-problems-python/

[ nan 0.18961404] => nan
[ 0.18961404 0.25956078] => 0.189614044109

sequence = array(sequence)  去掉值
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	# remove rows with missing values
	df.dropna(inplace=True)
	values = df.values
	# specify input and output data
	X, y = values, values[:, 0]

sequence = array(sequence)   替换值
	# create lag
	df = DataFrame(sequence)
	df = concat([df.shift(1), df], axis=1)
	# replace missing values with -1
	df.fillna(-1, inplace=True)
	values = df.values
	# specify input and output data
	X, y = values, values[:, 1]

正常层
# define model
model = Sequential()
model.add(LSTM(5, input_shape=(2, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit model
for i in range(500):
	X, y = generate_data(n_timesteps)
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
添加mark层
# define model
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(2, 1)))
model.add(LSTM(5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

2.pad_sequences 和truncating 的使用


sequences = [
[1, 2, 3, 4],
[1, 2, 3],
[1]
]
The default padding value is 0.0, which is suitable for most applications, although this can be changed
 
by specifying the preferred value via the “value” argument. For example:
pad_sequences(..., value=99)
Pre-sequence padding is the default (padding=’pre’)

The example below demonstrates pre-padding 3-input sequences with 0 values.
from keras.preprocessing.sequence import pad_sequences
# define sequences
sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
# pad sequence
padded = pad_sequences(sequences)
print(padded)

[[1 2 3 4]
[0 1 2 3]

padded = pad_sequences(sequences, padding='post')
print(padded)


[[1 2 3 4]
[1 2 3 0]
[1 0 0 0]]
# pad sequence
padded = pad_sequences(sequences, maxlen=5) 扩充
print(padded)

[[0 1 2 3 4]
[0 0 1 2 3]
[0 0 0 0 1]]

sequences = [
	[1, 2, 3, 4],
	   [1, 2, 3],
		     [1]
	]
# truncate sequence   截断
truncated= pad_sequences(sequences, maxlen=2)
print(truncated)


[[3 4]
[2 3]
[0 1]]
# truncate sequence
truncated= pad_sequences(sequences, maxlen=2, truncating='post')
print(truncated)


[[1 2]
[1 2]
[0 1]]


3 随机种子
http://machinelearningmastery.com/reproducible-results-neural-networks-keras/
Randomness in Initialization, such as weights.
Randomness in Regularization, such as dropout.
Randomness in Layers, such as word embedding.
Randomness in Optimization, such as stochastic optimization.

四句代码
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


It is possible that there are other sources of randomness that you have not accounted for.

Randomness from a Third-Party Library

Perhaps your code is using an additional library that uses a different random number generator that too must be seeded.

Try cutting your code back to the minimum required (e.g. one data sample, one training epoch, etc.) and carefully read the API documentation in an effort to narrow down additional third-party libraries introducing randomness.

Randomness from Using the GPU

All of the above examples assume the code was run on a CPU.

array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
x[5::-2]  # reversed every other from index 5
array([5, 3, 1])

array([[12,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6,  7,  7]])
x2[:3, ::2]
array([[12,  2],
       [ 7,  8],
       [ 1,  7]])

x2[::-1, ::-1]
array([[ 7,  7,  6,  1],
       [ 8,  8,  6,  7],
       [ 4,  2,  5, 12]])

print(x2[:, 0])  # first column of x2
[12  7  1] 一个括号

print(x2[0])  # equivalent to x2[0, :]
[12  5  2  4]  二维获取可以省略后面的

http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.02-The-Basics-Of-NumPy-Arrays.ipynb
Subarrays as no-copy views
One important–and extremely useful–thing to know about array slices is that they return views rather than copies of the array data.
 This is one area in which NumPy array slicing differs from Python list slicing: in lists, slices will be copies.
 Consider our two-dimensional array from before:

Note that for this to work, the size of the initial array must match the size of the reshaped array.
 Where possible, the reshape method will use a no-copy view of the initial array, but with non-contiguous memory buffers this is not always the case.

x = np.array([1, 2, 3])
x.reshape((1, 3))==x[np.newaxis, :]
array([[1, 2, 3]])

x.reshape((3, 1))
x[:, np.newaxis]
array([[1],
       [2],
       [3]])


按照坐标轴拼接 0 行扩展，1 是列扩展
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
# concatenate along the first axis
np.concatenate([grid, grid])
array([[1, 2, 3],
       [4, 5, 6],
       [1, 2, 3],
       [4, 5, 6]])

# concatenate along the second axis (zero-indexed)
np.concatenate([grid, grid], axis=1)
array([[1, 2, 3, 1, 2, 3],
       [4, 5, 6, 4, 5, 6]])

x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, grid])  垂直加行 水平加列
array([[1, 2, 3],
       [9, 8, 7],
       [6, 5, 4]])

# horizontally stack the arrays
y = np.array([[99],
              [99]])
np.hstack([grid, y])
array([[ 9,  8,  7, 99],
       [ 6,  5,  4, 99]])

分割数组
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)

[1 2 3] [99 99] [3 2 1]


array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])

upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)
[[0 1 2 3]
 [4 5 6 7]]
[[ 8  9 10 11]
 [12 13 14 15]]


left, right = np.hsplit(grid, [2])
print(left)
print(right)
[[ 0  1]
 [ 4  5]
 [ 8  9]
 [12 13]]
[[ 2  3]
 [ 6  7]
 [10 11]
 [14 15]]
