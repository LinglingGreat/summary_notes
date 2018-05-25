以前的方法只适用于小型网络。

Although a few very recent studies approach the embedding of large-scale networks, these methods either
use an indirect approach that is not designed for networks or lack a clear objective function tailored for network embedding。

LINE, which is able to scale to very large, arbitrary types of networks: undirected, directed and/or weighted

The model optimizes an objective which preserves both the local and global network structures

两种相似性：

first-order：两个点之间有很强的联系(比如边的权重很大)。

second-order：两个点有很多相同的邻居

they should be represented closely to each other in the embedded space

**LINE with First-order Proximity**

对于无向边(i, j)，$v_i和v_j$的联合概率是$p_1(v_i, v_j) = \frac1{1+exp(-\overrightarrow{u_i}^T · \overrightarrow{u_j})}$

相应的经验概率是：$\hat p_1(i,j) = \frac{w_{ij}}W, W=\sum_{(i,j) \in E}w_{ij}$

最小化两个概率分布之间的距离，用KL散度距离，因此目标函数是：$O_1 = d(\hat p_1(·,·),p_1(·,·)) = -\sum_{(i,j) \in E}w_{ij}logp_1(v_i, v_j)$

**LINE with Second-order Proximity**

无向和有向图都适用

每个点被看作是"context"，在"context"上有相似分布的结点视作相似。

每个点有两个角色：the vertex itself and a specific "context" of other vertices，对应两个向量$\overrightarrow{u_i}和\overrightarrow{u_i}'$

对于每个有向边(i,j), 定义由点$v_i$ 生成的"context" $v_j$ 的概率为：$p_2(v_j | v_i) = \frac{exp(\overrightarrow{u_j}'^T · \overrightarrow{u_i})}{\sum_{k=1}^{|V|}exp(\overrightarrow{u_k}'^T · \overrightarrow{u_i})}$

where |V| is the number of vertices or "contexts."

最小化目标函数：

$O_2 = \sum_{i \in V}\lambda_i d(\hat p_2(·|v_i),p_2(·|v_i)) = -\sum_{(i,j) \in E}w_{ij}logp_2(v_j|v_i)$

其中$\lambda_i$ 是the prestige声望 of vertex i in the network, which can be measured by the degree or estimated through algorithms such as PageRank，这里取$d_i=\sum_{k \in N(i)}w_{ik}$ 即点i的出度，$\hat p_2(v_j | v_i) = \frac{w_{ij}}{d_i}$ 

分别训练两个LINE模型，然后将两个模型的每个点的向量拼接起来，也可以联合训练两个目标函数(future work)

优化目标函数$O_2$ is computationally expensive,采用负采样方法。对每条边(i,j), 目标函数为：

$log\sigma(\overrightarrow{u_j}'^T · \overrightarrow{u_i})+\sum_{i=1}^K E_{v_n \sim P_n(v)}[log\sigma(\overrightarrow{u_n}'^T · \overrightarrow{u_i})]$

set $ P_n(v) \propto d_v^{3/4}, d_v 是点v的出度$

用asynchronous stochastic gradient algorithm(ASGD)优化上述方程。

In each step, the ASGD algorithm samples a mini-batch of edges and then updates the model parameters.

梯度会由权重乘积而来，而权重的方差很大，会导致难以找到合适的学习率。

用the alias table method根据边的权重进行采样，将采样到的边视为binary edges.

对于目标函数$O_1$也采用负采样的方法，将上式中的$\overrightarrow{u_j}'^T 换成\overrightarrow{u_j}^T$ 

实验：a language network, two social networks, and two citation networks

second-order proximity suffers when the network is extremely sparse, and it outperforms rst-order proximity when there are sufficient nodes in the neighborhood of a node

second-order proximity does not work well for nodes with a low degree



Tang J, Qu M, Wang M, et al. 

Line: Large-scale information network embedding

[C]//Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2015: 1067-1077. 

DeepWalk之后提出