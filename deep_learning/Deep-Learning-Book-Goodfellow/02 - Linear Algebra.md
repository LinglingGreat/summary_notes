# 深度学习

英文版深度学习：<http://www.deeplearningbook.org/> 

中文翻译版深度学习：<https://github.com/exacity/deeplearningbook-chinese>

参考资料：https://github.com/InveterateLearner/Deep-Learning-Book-Chapter-Summaries

https://medium.com/inveterate-learner

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

$  Ax = B \tag{2.11} $

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

如果逆矩阵存在，那么上式肯定对于每一个向量$b$恰好存在一个解。

### 2.4 线性相关和生成子空间

对于方程组而言，对于向量$b$的某些值，有可能不存在解，或者存在无限多个解。存在多于一个解但是少于无限多个解的情况是不可能发生的。因为如果$x$和$y$都是某方程组的解，则$z=\alpha x+(1-\alpha)y$也是该方程组的解。

为了分析方程有多少个解，我们可以将A 的列向量看作从原点（origin）（元素都是零的向量）出发的不同方向，确定有多少种方法可以到达向量b。在这个观点下，向量$x$ 中的每个元素表示我们应该沿着这些方向走多远，即$x_i $表示我们需要沿着第$i $个向量的方向走多远：

$$ Ax = \sum_{i=1}^{n} x_iA_{:, i}  $$

一般而言，这种操作被称为**线性组合**（linear combination）。形式上，一组向量的线
性组合，是指每个向量乘以对应标量系数之后的和，即：

$\sum_{i=1}^{n} c_iv^{(i)}$

一组向量的**生成子空间**（span）是原始向量线性组合后所能抵达的点的集合。

确定$Ax = b$ 是否有解相当于确定向量$b$ 是否在$A$ 列向量的生成子空间中。这个特殊的生成子空间被称为$A $的**列空间**（column space）或者$A $的**值域**（range）。

为了使方程$Ax = b$ 对于任意向量$b \in R^m$ 都存在解，我们要求$A$ 的列空间构成整个$R^m$。如果$R^m$中的某个点不在$A$ 的列空间中，那么该点对应的$b$ 会使得该方程没有解。矩阵A 的列空间是整个$R^m$ 的要求，意味着$A$ 至少有m 列，即n ≥ m。否则，$A$ 列空间的维数会小于m。例如，假设A 是一个3 x 2 的矩阵。目标$b$ 是3 维的，但是$x$ 只有2 维。所以无论如何修改$x$ 的值，也只能描绘出$R^3$空间中的二维平面。当且仅当向量$b$ 在该二维平面中时，该方程有解。

不等式n ≥  m 仅是方程对每一点都有解的必要条件。这不是一个充分条件，因为有些列向量可能是冗余的。

正式地说，这种冗余被称为**线性相关**（linear dependence）。如果一组向量中的任意一个向量都不能表示成其他向量的线性组合，那么这组向量称为**线性无关**（linearly independent）.

如果一个矩阵的列空间涵盖整个$R^m$，那么该矩阵必须包含至少一组m 个线性无关的向量。
这是式(2.11) 对于每一个向量b 的取值都有解的充分必要条件。值得注意的是，这个条件是说该向量集恰好有m 个线性无关的列向量，而不是至少m 个。

要想使矩阵可逆，我们还需要保证式(2.11) 对于每一个b 值至多有一个解。为此，我们需要确保该矩阵至多有m 个列向量。否则，该方程会有不止一个解。综上所述，这意味着该矩阵必须是一个**方阵**（square），即m = n，并且所有列向量都是线性无关的。一个列向量线性相关的方阵被称为**奇异的**（singular）。如果矩阵A 不是一个方阵或者是一个奇异的方阵，该方程仍然可能有解。但是我们不能使用矩阵逆去求解。

对于方阵而言，它的左逆和右逆是相等的。

### 2.5 范数

有时我们需要衡量一个向量的大小。在机器学习中，我们经常使用被称为**范数**（norm）的函数衡量向量大小。形式上，$L^p $范数定义如下

$$ ||\mathbf{x}||_p = (\sum_{i} |x_i|^p)^{\frac{1}{p}} $$

其中$p \in R, p ≥ 1$。

范数是将向量映射到非负值的函数。直观上来说，向量$x $的范数衡量从原点到点$x$ 的距离。更严格地说，范数是满足下列性质的任意函数：

- $f(x) = 0 \Rightarrow x = 0$
- $f(x+y) \leq f(x) + f(y)$ (三角不等式 **triangle inequality**)
- $\forall \alpha \in ℝ, \hspace{.1cm} f(\alpha x) = |\alpha|f(x)$

不同类型的范数：

- **$L_2 $范数**。当p = 2 时，$L_2 $范数被称为**欧几里得范数**（Euclidean norm）。它表示从原点出发到向量x 确定的点的欧几里得距离。$L_2 $范数在机器学习中出现地十分频繁，经常简化表示为$||x||$，略去了下标2。平方$L_2 $ 范数也经常用来衡量向量的大小，可以简单地通过点积$x^Tx$ 计算。

平方$L_2 $范数在数学和计算上都比$L_2 $范数本身更方便。例如，平方$L_2 $范数对x 中每个元素的导数只取决于对应的元素，而$L_2 $范数对每个元素的导数却和整个向量相关。但是在很多情况下，平方$L_2 $范数也可能不受欢迎，因为它在原点附近增长得十分缓慢。

- **$L_1$范数**。在某些机器学习应用中，区分恰好是零的元素和非零但值很小的元素是很重要的。在这些情况下，我们转而使用在各个位置斜率相同，同时保持简单的数学形式的函数：$L_1$范数。$L_1$范数可以简化如下：

$ ||x||_1 = \sum_i |x_i|$

当机器学习问题中零和非零元素之间的差异非常重要时，通常会使用L1 范数。每当x 中某个元素从0 增加ϵ，对应的L1 范数也会增加ϵ。

- **最大范数（max norm）或者$L^{\infty}$范数**。这个范数表示向量中具有最大幅值的元素的绝对值：

$||x||_{\infty} = \displaystyle \max_{i}|x_i|$

- **Frobenius 范数（Frobenius norm）**。衡量矩阵的大小，类似于向量的$L^2$范数：

$||A||_F = \sqrt{\displaystyle \sum_{i,j} A_{i,j}^2} $

两个向量的点积（dot product）可以用范数来表示。具体地，

$x^⊤y = ||x||_2||y||_2 cos\theta$

其中$\theta$表示$x$ 和$y$ 之间的夹角。

### 2.6 特殊类型的矩阵和向量

**对角矩阵**（diagonal matrix）只在主对角线上含有非零元素，其他位置都是零。

- 我们用diag(v) 表示一个对角元素由向量v 中元素给定的对角方阵。
- 对角矩阵的乘法计算很高效。计算乘法$diag(v)x$，我们只需要将x 中的每个元素$x_i$ 放大$v_i$ 倍。换言之，$diag(v)x = v ⊙ x$。
- 计算对角方阵的逆矩阵也很高效。对角方阵的逆矩阵存在，当且仅当对角元素都是非零值，在这种情况下，$diag(v)^{-1 }= diag([1/v_1,...,1/v_n]^⊤)$。
- 在很多情况下，我们可以根据任意矩阵导出一些通用的机器学习算法；但通过将一些矩阵限制为对角矩阵，我们可以得到计算代价较低的（并且简明扼要的）算法。
- 不是所有的对角矩阵都是方阵。长方形的矩阵也有可能是对角矩阵。非方阵的对角矩阵没有逆矩阵，但我们仍然可以高效地计算它们的乘法。对于一个长方形对角矩阵D 而言，乘法$Dx$ 会涉及到x 中每个元素的缩放，如果D 是瘦长型矩阵，那么在缩放后的末尾添加一些零；如果D 是胖宽型矩阵，那么在缩放后去掉最后一些元素。

**对称**（symmetric）矩阵是转置和自己相等的矩阵：$A=A^T$

当某些不依赖参数顺序的双参数函数生成元素时，对称矩阵经常会出现。例如距离度量矩阵。

单位向量（unit vector）是具有单位范数（unit norm）的向量：$||x||_2=1$

**正交**：如果$x^⊤y$ = 0，那么向量x 和向量y 互相正交（orthogonal）。如果两个向量都有非零范数，那么这两个向量之间的夹角是90 度。在$R^n$ 中，至多有n 个范数非零向量互相正交。如果这些向量不仅互相正交，并且范数都为1，那么我们称它们是**标准正交**（orthonormal）。

**正交矩阵**（orthogonal matrix）是指行向量和列向量是分别标准正交的方阵：

$$ A^TA = AA^T = I \Rightarrow A^{-1} = A^T $$

正交矩阵的求逆代价小。

### 2.7 特征分解

正如我们可以通过分解质因数来发现整数的一些内在性质（例如12 = 2 x 2 x 3，发现12的倍数可以被3整除，12不能被5整除），我们也可以通过分解矩阵来发现矩阵表示成数组元素时不明显的函数性质。

**特征分解**（eigendecomposition）是使用最广的矩阵分解之一，即我们将矩阵分解成一组特征向量和特征值。

方阵A 的**特征向量**（eigenvector）是指与A 相乘后相当于对该向量进行缩放的非零向量v：

$$ Av = \lambda v $$

标量$\lambda$被称为这个特征向量对应的**特征值**（eigenvalue）。（类似地，我们也可以定义左特征向量（left eigenvector）$v^⊤A = \lambda v^⊤$，但是通常我们更关注右特征向量（right eigenvector））。

如果v 是A 的特征向量，那么任何缩放后的向量sv ($s \in R，s \neq 0$) 也是A 的特征向量。此外，sv 和v 有相同的特征值。基于这个原因，通常我们只考虑**单位特征向量**。

假设矩阵A 有n 个线性无关的特征向量连接成的矩阵（每一列是一个特征向量） $V = [v^{(1)}, ... , v^{(n)}]$ ，相应地特征值组成的向量 $\lambda = [\lambda_1, ... , \lambda_n]$ ，因此A 的**特征分解**（eigendecomposition）可以记作:

$$ A = Vdiag(\lambda)V^{-1} $$

构建具有特定特征值和特征向量的矩阵，能够使我们在目标方向上延伸空间。

将矩阵分解（decompose）成特征值和特征向量，可以帮助我们分析矩阵的特定性质。

不是每一个矩阵都可以分解成特征值和特征向量。在某些情况下，特征分解存
在，但是会涉及复数而非实数。

每个实对称矩阵都可以分解成实特征向量和实特征值，即

$A=Q \Lambda Q^T$

其中Q 是A 的特征向量组成的正交矩阵，$\Lambda$是对角矩阵。因为Q 是正交矩阵，我们可以将$A$ 看作沿方向$v^{(i)}$延展 $\lambda_i$ 倍的空间 . 

虽然任意一个实对称矩阵A 都有特征分解，但是特征分解可能并不唯一。如果两个或多个特征向量拥有相同的特征值，那么在由这些特征向量产生的生成子空间中，任意一组正交向量都是该特征值对应的特征向量。因此，我们可以等价地从这些特征向量中构成Q 作为替代。

我们通常降序排列对角矩阵的元素$\lambda$。在该约定下，特征分解唯一当且仅当所有的特征值都是唯一的。

矩阵的特征分解给了我们很多关于矩阵的有用信息。**矩阵是奇异的当且仅当含有零特征值。**实对称矩阵的特征分解也可以用于优化二次方程$f(x) = x^⊤Ax$，其中限制$||x||_2 = 1$。当x 等于A 的某个特征向量时，f 将返回对应的特征值。在限制条件下，函数f 的最大值是最大特征值，最小值是最小特征值。

两种有用的矩阵:

- **正定Positive definite**: 所有特征值都是正数的矩阵，有性质$x^TAx = 0 \Rightarrow x = 0$.
- **半正定Positive semidefinite**: 所有特征值都是非负数的矩阵，有性质$\forall x, \hspace{0.1cm} x^TAx \geq 0$.

同样地，所有特征值都是负数的矩阵被称为负定（negative definite）；所有特征值都是非正数的矩阵被称为半负定（negative semidefinite）。

```
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

A = np.array([[1, 2], [4, 3]])
w, v = np.linalg.eig(A)    # 分别是特征值矩阵，特征向量矩阵

def normalize(u):
    for i, x in enumerate(u):
        norm = np.sqrt(sum([val**2 for val in x]))   # 计算范数
        for j, y in enumerate(x):
            u[i][j] = y / norm   # 归一化

normalize(v)   # 将特征向量归一化

x_values = np.linspace(-1, 1, 1000)
y_values = np.array([np.sqrt(1 - (x**2)) for x in x_values])

x_values = np.concatenate([x_values, x_values])
y_values = np.concatenate([y_values, -y_values])   # 单位圆的数据

u = np.array([[x, y] for x, y in zip(x_values, y_values)])

trans = np.dot(A, u.T).T   # Au

fig = plt.figure()
arrow_dir = np.array([[0, 0, v[0][0], v[0][1]], [0, 0, v[1][0], v[1][1]]])    # 两个特征向量
X, Y, U, V = zip(*arrow_dir)
ax = fig.add_subplot(1, 2, 1)
plt.plot(x_values, y_values, '.')
ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
ax.axis('equal')

plt.xlim([-2, 2])
plt.ylim([-2, 2])


ax = fig.add_subplot(1, 2, 2)
ax.axis('equal')
plt.plot(trans[:, 0], trans[:, 1], '.')

plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.show()
```

![eigendecomposition](img/eigendecomposition.png)

左边是一个单位圆u，右边是所有$Au$点的集合，A拉伸了单位圆。

### 2.8 奇异值分解

**SVD** is another way of factorizing a matrix to give **singular values** and **singular vectors**. However, it is more generally applicable than eigendecomposition, e.g. eigendecomposition is not defined for a non-square matrix and we opt for SVD there. Here, we write $A$ as:

$$A = UDV^T $$

**Shapes**:
- $A$: $m$ x $n$
- $U$: $m$ x $m$
- $D$: $m$ x $n$
- $V$: $n$ x $n$

**Properties**:
- $U$ and $V$ are defined to be orthogonal matrices. 
- $D$ is a diagonal matrix (not necessarily square), diagonal elements of which are called **singular values**.
- The columns of $U$ are known as **left-singular vectors** (eigenvectors of $AA^T$) and those of $V$ are called **right-singular vectors** (eigenvectors of $A^TA$).
- Most useful feature is to extend matrix inversion to non-square matrices.



### 2.9 Moore-Penrose 伪逆

Suppose we want a left-inverse $B$ of a matrix $A$ ($m$ x $n$)to solve a linear equation:
$$ Ax = y \Rightarrow x = By$$ 

We define the pseudoinverse of $A$ as:

$$A^+ = \lim\limits_{\alpha \rightarrow 0} (A^TA + \alpha I)^{-1}A^T$$

However, for practical algorithms its defined as:

$$ A^+ = VD^+U^T $$

where $U$, $D$ and $V$ are the SVD of $A$ and $D^+$ is obtained by taking the reciprocal of all non-zero elements of D and then taking the transpose of the resulting matrix.

**Case 1**: m <= n

Using $A^+$, gives one of many possible solutions, with the minimal **Euclidean norm**:

$$ x = A^{+}y $$

**Case 2**: m > n

It is possible for there to be no solution and $A^+$ gives the $x$ such that $Ax$ is as close to $y$ in terms of the **Euclidean norm** $||Ax - y||$.

### 2.10 迹运算

The trace operator gives the sum of all the diagonal elements.

$$ Tr(A) = \sum_{i}A_{i,i}$$

Properties:

- $||A||_F = \sqrt{Tr(AA^T)} $ (**Frobenius Norm**)
- $Tr(A) = Tr(A^T)$ (**Transpose Invariance**)
- $Tr(ABC) = Tr(CAB) = Tr(BCA)$ (**Cyclical Invariance** given that the individual matrix multiplications are defined)



### 2.11行列式

The determinant of a square matrix (denoted by $det(A)$) maps matrices to real scalars. It is equal to the product of all the eigenvalues of the matrix. It denotes how much multiplication by the matrix expands or contracts space. If the value is 0, then space is contracted completely atleast along one dimension causing it to lose all its volume. If the value is 1, then the transformation preserves volume.

### 2.12 实例：主成分分析

