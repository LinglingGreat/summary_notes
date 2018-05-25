# asymmetric proximity preserving(APP)

we propose an asymmetric proximity preserving(APP) graph embedding method via random walk with restart, which captures both asymmetric and high-order similarities between node pairs. We give theoretical analysis that our method implicitly preserves the Rooted PageRank score for any two vertices.

考虑节点间的非对称相似度，每个节点训练出两个向量：头向量和尾向量。两个节点的相似性用两个节点的头向量和尾向量的内积表示

**Higher order proximity :**

**1.SimRank**

SimRank 是一种基于图的拓扑结构来衡量图中任意两个点的相似程度的方法。 在基于链接的相似性度量领域中SimRank被认为与PageRank在信息检索领域具有同样重要的地位 。

如果两个点在图中的邻域比较相似（有很多相似邻居），则这两个点也应该比较相似。即两个点是否相似，由他们的邻居是否相似来决定。而他们的邻居是否相似又由他们邻居的邻居的相似性决定。

跟pagerank类似，这也是一个迭代的定义。即通过迭代的方式计算两个点之间的相似度，最终取收敛的相似度。

**如果两个节点相同，则相似度是1。如果两个节点不同，那他们的相似度就等于他们两个所有一步邻居的两两相似度的均值，再乘以衰减系数cc。 **

SimRank的特点：完全基于结构信息，且可以计算图中任意两个节点间的相似度。  

Jeh, G., and Widom, J. 2002. 

Simrank: a measure of structural-context similarity. 

In International Conference on Knowledge Discovery and Data Mining, 538–543. ACM.



**2.Rooted PageRank**

PageRank的计算充分利用了两个假设：数量假设和质量假设。步骤如下：
      **1）在初始阶段：**网页通过链接关系构建起Web图，每个页面设置相同的PageRank值，通过若干轮的计算，会得到每个页面所获得的最终PageRank值。随着每一轮的计算进行，网页当前的PageRank值会不断得到更新。

​      **2）在一轮中更新页面PageRank得分的计算方法：**在一轮更新页面PageRank得分的计算中，每个页面将其当前的PageRank值平均分配到本页面包含的出链上，这样每个链接即获得了相应的权值。而每个页面将所有指向本页面的入链所传入的权值求和，即可得到新的PageRank得分。当每个页面都获得了更新后的PageRank值，就完成了一轮PageRank计算。

**PageRank的迭代公式：$R=q×P*R+(1-q)*e/N, e是单位向量$ **

**主题敏感PageRank中：$R=q×P*R+(1-q)*s/N$ , s是这样一个向量：对于某topic的s，如果网页k在此topic中，则s中第k个元素为1，否则为0。对于每个topic都有一个不同的s，而|s|表示s中1的数量。每个网页归到一个topic。**

在PageRank中e/N是一个均匀分布，而在PPR中则根据用户的preference指定权重，例如用户指定了10个页面，则可以设置这10个页面对应的权重均为1/10，其余均为0。 

一般来说用户会对某些领域感兴趣，同时，当浏览某个页面时，这个页面也是与某个主题相关的（比如体育报道或者娱乐新闻），所以，当用户看完当前页面，希望跳转时，更倾向于点击和当前页面主题类似的链接，即主题敏感PageRank是将用户兴趣、页面主题以及链接所指向网页与当前网页主题的相似程度综合考虑而建立的模型。 

PageRank是全局性的网页重要性衡量标准，每个网页会根据链接情况，被赋予一个唯一的PageRank分值。主题敏感PageRank在此点有所不同，该算法引入16种主题类型，对于某个网页来说，对应某个主题类型都有相应的PageRank分值，即每个网页会被赋予16个主题相关PageRank分值。 

Haveliwala, T. H. 2002. 

Topic-sensitive pagerank. 

In Proceedings of the 11th international conference on World Wide Web



**3.Katz**

https://en.wikipedia.org/wiki/Katz_centrality

两个节点之间所有路径的加权和

Katz, L. 1953. 

A new status index derived from sociometricanalysis. 

Psychometrika



**Asymmetric graph embedding**

High-Order Proximity preserved Embedding(HOPE for short)

we first derive a general formulation of a class of high-order proximity measurements, then apply generalized SVD to the general formulation, whose time complexity is linear with the size of graph.

Ou, M.; Cui, P.; Pei, J.; and Zhu,W. 2016. 

Asymmetric transitivity preserving graph embedding. 

In International Conference on Knowledge Discovery and Data Mining. ACM.



**the Monte-Carlo End-Point sampling method **

Let $A$ denote the adjacency matrix of the web graph with normalized rows and $c ∈ (0, 1)$ the teleportation probability. In addition, let $r$ be the so-called preference vector inducing a probability distribution over $V$. PageRank vector $p$ is defined as the solution of the following equation

p = (1 - c)  · pA + c ·  r

要么以概率c ·  r选择该网页，要么从别的网页跳到该网页来。

按上述式子迭代的话复杂度很高，所以一般用蒙特卡洛模拟。

To compute a rooted pagerank vector for v, the Monte Carlo approach randomly samples N independent paths started from v, with stoping probability of c. Then the rooted pagerank value can be approximated as,
$ppr_v(u) = |PathEndsAt(u) |/ N$

网页v对于主体u的pagerank score

论文中用蒙特卡洛方法模拟v到达u的概率，用于目标函数中。

Fogaras, D.; R´acz, B.; Csalog´any, K.; and Sarl´os, T. 2005.

Towards scaling fully personalized pagerank: Algorithms, lower bounds, and experiments.

Internet Mathematics



**DeepWalk**



Perozzi, B.; Al-Rfou, R.; and Skiena, S. 2014. 

Deepwalk: Online learning of social representations. 

In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining

**Line**



Tang, J.; Qu, M.;Wang, M.; Zhang, M.; Yan, J.; and Mei, Q. 2015

Line: Large-scale information network embedding. 

In Proceedings of the 24th International Conference on World Wide Web

**Node2Vec**



Grover, A., and Leskovec, J. 2016. 

node2vec: Scalable feature learning for networks. 

In International Conference on Knowledge Discovery and Data Mining. ACM.



**Common Neighbors (CNbrs for short)**

最简单的基于局部信息的相似性方法。如果两个节点间的共同邻居结点越多，那么两者存在链接的可能性就越大。得分公式：

$score(u,v)=|N(u)\bigcap N(v)|$

**Adamic Adar (Adar for short) **

起初用于计算两个用户的个人主页的相似性。在计算两个用户个人主页时，首先要提取两主页的公共关键词，然后计算每个公共关键词的权重，最后对所有的公共关键词进行加权求和。关键词的权重与关键词出现的次数的倒数成反比。

$\sum_{t\in N(u)\bigcap N(v)} \frac1{log|N(t)|}$

**Jaccard Coefficience **

利用两节点共同邻居的交集与并集个数之比，定义为两节点的相似度。

$score(u,v) = |N(u)\bigcap N(v)|/ |N(u)\bigcup N(v)|$



Scalable Graph Embedding for Asymmetric Proximity    C Zhou，Y Liu，X Liu，... - 2017