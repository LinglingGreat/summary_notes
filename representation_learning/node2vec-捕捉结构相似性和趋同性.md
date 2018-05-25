a semi-supervised algorithm for scalable feature learning in networks

简单来说就是将原有社交网络中的图结构，表达成特征向量矩阵，每一个node（可以是人、物品、内容等）表示成一个特征向量，用向量与向量之间的矩阵运算来得到相互的关系。 (如向量均值，Hadamard积，Weighted-L1, Weighted-L2)

there is no clear winning sampling strategy that works across all networks and all prediction tasks. This is a major shortcoming of prior work which fail to offer any flexibility in sampling of nodes from a network。

Our algorithm node2vec overcomes this limitation by designing a flexible objective that is not tied to a particular sampling strategy and provides parameters to tune the explored search space。

**Algorithm 1 The node2vec algorithm.**
**LearnFeatures** (Graph G = (V, E, W), Dimensions d, Walks per node r, Walk length l, Context size k, Return p, In-out q)
$\pi$ = PreprocessModifiedWeights(G, p, q)
	$G' = (V, E,  \pi)$
	Initialize walks to Empty
	for iter = 1 to r do
		for all nodes $u \in V$ do
			walk = node2vecWalk$(G', u, l)$
			Append walk to walks
	f = StochasticGradientDescent(k, d, walks)
	return f
**node2vecWalk** (Graph $G' = (V, E,  \pi)$, Start node u, Length l)
	Inititalize walk to [u]
	for walk_iter = 1 to l do
		curr = walk[-1]
		$V_{curr}$ = GetNeighbors(curr, G')
		s = AliasSample$(V_{curr}, \pi)$
		Append s to walk
	return walk

RandomWalks:

$P(c_i=x | c_{i-1}=v) = \frac{\pi_{vx}}Z \; if (v,x) \in E  \; otherwise \; 0$

$\pi_{vx} = \alpha_{pq}(t, x) · w_{vx}$ 

$\alpha_{pq}(t, x) = \frac1p    \; if \; d_{tx} =0$

$\alpha_{pq}(t, x) = 1    \; if \; d_{tx} =1$

$\alpha_{pq}(t, x) = \frac1q    \; if \; d_{tx} =2$

$d_{tx}$ denotes the shortest path distance between nodes t and x. t是当前点的上一个点，v是当前点，x是v的邻居结点

Parameter p controls the likelihood of immediately revisiting a node in the walk.

Setting it to a high value(> max(q; 1)) ensures that we are less likely to sample an already visited node in the following two steps (unless the next node in the walk had no other neighbor).

if q > 1, 倾向于BFS广度优先搜索，if q < 1, 倾向于DFS深度优先搜索



极大似然优化，目标函数是：

$max_f \sum_{u \in V} log Pr(N_S(u) | f(u))$

$$Pr(N_S(u) | f(u)) = \prod _{n_i \in N_S(u)} Pr(n_i | f(u))$$

$Pr(n_i | f(u)) = \frac{exp(f(n_i)·f(u))}{\sum_{v \in V}exp(f(v)·f(u))}$

因此目标函数变为：

$max_f \sum_{u \in V}[-logZ_u + \sum_{n_i \in N_S(u)} f(n_i)·f(u)]$

$Z_u = \sum_{v \in V} exp(f(u)·f(v))$

可以通过负采样来优化分母的计算量 ,

用随机梯度下降法优化上述目标函数



实验：聚类，多标签的分类，链接预测

参数敏感分析，扰动分析，可扩展性

- 用学到的向量去做分类任务的特征，结果比其他方法好很多，并且这种方法很鲁棒！即使缺少边也没问题。
- 可扩展到大规模 node！

Grover A, Leskovec J. 

node2vec: Scalable feature learning for networks

[C]//Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2016