只适用于不加权的图

算法1：$DeepWalk(G, w, d, \gamma, t)$

**Input: **

graph G(V;E)
window size w
embedding size d
walks per vertex  $\gamma$
walk length t
**Output**: matrix of vertex representations $\Phi \in R^{|V|×d}$
1:  Initialization: Sample $\Phi$ from $U^{|V|×d}$
2:  Build a binary Tree T from V
3:  for i = 0 to  $\gamma$ do
4: 	O = Shuffle(V)
5: 	for each $v_i \in O$ do
6: 		$W_{v_i} = RandomWalk(G, v_i, t)$
7: 		SkipGram($\Phi, W_{v_i} , w$)
8: 	end for
9:  end for

算法2：SkipGram($\Phi, W_{v_i} , w$)

1:  for each $v_j \in W_{v_i}$ do
2:  	for each $u_k \in W_{v_i} [j - w : j + w]$ do
3:  		$J(\Phi) = -log Pr(u_k| \Phi (v_j))$
4: 		$\Phi = \Phi - \alpha * \frac{\partial J}{\partial \Phi}$                        (SGD)
5: 	end for
6:  end for



Computing the partition function (normalization factor) is expensive, so instead we will factorize the conditional probability using Hierarchical softmax. We assign the vertices to the leaves of a binary tree, turning the prediction problem into maximizing the probability of a specic path in the hierarchy.If the path to vertex $u_k$ is identied by a sequence of tree nodes $(b_0, b_1, ..., b_{\lceil log|V |\rceil}), (b_0 = root, b_{\lceil log|V |\rceil} = u_k)$ then

$Pr(u_k| \Phi (v_j)) = \prod_{l=1}^{\lceil log|V |\rceil} Pr(b_l| \Phi (v_j))$

$Pr(b_l| \Phi (v_j))$ could be modeled by a binary classifier that is assigned to the parent of the node $b_l$ as

$Pr(b_l| \Phi (v_j)) =  1/(1+e^{-\Phi(v_j)· \Psi(b_l)})$

where $\Psi(b_l) \in R^d$ is the representation assigned to tree node $b_l$'s parent



实验：多标签分类

参数敏感性分析

Perozzi, B.; Al-Rfou, R.; and Skiena, S. 2014. 

Deepwalk: Online learning of social representations. 

In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining