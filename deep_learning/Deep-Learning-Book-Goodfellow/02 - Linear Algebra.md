# 深度学习

英文版深度学习：<http://www.deeplearningbook.org/> 

中文翻译版深度学习：<https://github.com/exacity/deeplearningbook-chinese>

## 第1部分 应用数学与机器学习基础

## 第2章 线性代数

### 2.1 标量、向量、矩阵和张量

**标量**(scalar)：一个单独的数，用斜体表示，通常被赋予小写的变量名称。

**向量**(vector)：一列数，粗体的小写变量名称，比如$x$.

**矩阵**(matrix)：一个二维数组，粗体的大写变量名称，比如$A$.

**张量**(tensor)：一个数组中的元素分布在若干维坐标的规则网格中，

允许矩阵和向量相加，产生一个新的矩阵：

$C = A + b$ ，其中 $C_{i, j} = A_{i, j} + b_{j}$，即向量$b$和矩阵$A$的每一行相加。

这种隐式地复制向量$b$到很多位置的方式，称为**广播**(broadcasting)。

两个广播的例子：

```python
import numpy as np
M = np.arange(9).reshape(3, 3)
C = np.arange(3).reshape(3, 1)
print("M:")
print(M)
print("C:")
print(C)
print("M+C:")
print(M+C)
```

```
M:
[[0 1 2]
 [3 4 5]
 [6 7 8]]
C:
[[0]
 [1]
 [2]]
M+C:
[[ 0  1  2]
 [ 4  5  6]
 [ 8  9 10]]
```

```python
import numpy as np
M = np.arange(9).reshape(3, 3)
C = np.arange(3).reshape(1, 3)
print("M:")
print(M)
print("C:")
print(C)
print("M+C:")
print(M+C)
```

```
M:
[[0 1 2]
 [3 4 5]
 [6 7 8]]
C:
[[0 1 2]]
M+C:
[[ 0  2  4]
 [ 3  5  7]
 [ 6  8 10]]
```

### 2.2 矩阵和向量相乘

两个矩阵$A_{mn}$ 和 $B_{kp}$ 的**矩阵乘积**(matrix product)是$C$, 为了使乘法可被定义，必须满足 $n = k$.  $C (= AB)$ 的形状是 $m$ x $p$. 

**元素对应乘积**(element-wise product)或**Hadamard乘积**(Hadamard product)是两个矩阵对应元素的乘积，记为 $A \odot B$ . 

两个相同维数的向量$x$ 和 $y$ 的**点积**(dot-product) 可看作矩阵乘积$x^Ty$.

有用的性质:

1. $A(B+C) = AB + AC$ (分配律)
2. $A(BC) = (AB)C$    (结合律)
3. $AB \ne BA$ (一般情况下不满足交换律)
4. $(AB)^T = B^TA^T$
5. $x^Ty = (x^Ty)^T = y^Tx$

线性方程组:

$  Ax = B \tag{1} $

其中 $A \in ℝ^{mxn}$ 和 $b \in ℝ^{m}$ 都是已知的，$x \in ℝ^{n}$是要求解的未知向量。上式可以写成$m$个线性方程，其中 第 $i$ 个方程是：

$A_{i,1}x_1 + A_{i,2}x_2 + ... + A_{i,n}x_n = b_i$

### 2.3 单位矩阵和逆矩阵

**单位矩阵**(identity matrix)，任意向量和单位矩阵相乘，都不会改变。我们将保持n维向量不变的单位矩阵记作$I_n \in R^{n\times n}$，单位矩阵的所有沿主对角线的元素都是1，而其他位置的所有元素都是0.

矩阵$A$的逆矩阵记作$A^{-1}$，其定义的矩阵满足如下条件：

$A^{-1}A=I_n$

可以用以下步骤求解方程组：

$$  Ax = b \\ 
A^{-1}Ax = A^{-1}b \\
I_nx = A^{-1}b \\ 
x = A^{-1}b \\
$$

### 2.4 线性相关和生成子空间









