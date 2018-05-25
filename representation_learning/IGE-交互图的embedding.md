IGE (Interaction Graph Embedding)

以前的节点嵌入方法都集中在static non-attributed graphs上，现实中attributed interaction graphs.它包含两类实体（如投资者/股票），以及edges of temporal interactions with attributes (e.g. transactions and reviews) 

Our model consists of an attributes encoding network and two prediction networks. Three networks are trained simultaneously by back-propagation. Negative sampling method is applied in training prediction networks for acceleration.将属性转化成固定长度的向量的编码网络（处理属性的异质性）；利用属性向量预测两个实体的暂时事件的子网络 如 预测投资者事件的子网络；预测股票事件的子网络 s

**Attributed interaction graph:  **

G = (X,Y, E), X and Y are two disjoint sets of nodes, and E is the set of edges. Each edge e ∈ E can be represented as a tuple e = (x,y, t , a), where x ∈ X, y ∈ Y, t denotes the timestamp and a = (a1, . . . , am)
is the tuple of heterogeneous attributes of the edge. ai ∈ Ai is an instance of the i-th attribute.

例如(investor, stock, time, buy_or_sell, price, amount)

**Induced Edge List**

Given an attributed interaction graph G = (X,Y, E) and a source node x ∈ X, we say sx = (e1, . . . , en) is an edge list induced by the source node x, if every edge in sx is incident to x. Similarly, we can define a target node y’s induced edge list s′y .

**A Simplified Model**

Let G = (X,Y, E) be a given interaction graph where edges are without attributes. Inspired by the Skip- gram model and Paragraph Vector, we formulate the embedding learning as a maximum likelihood problem.

$L(\theta) = \alpha \sum_{x \in X} logP(s_x; \theta) + (1- \alpha) \sum_{y \in Y} logP(s_y’; \theta)$

$\alpha \in [0,1]$ 是超参数，is used to make a trade-off between the importance of source nodes induced lists and target nodes induced lists。

$\theta$ 代表所有的模型参数

$logP(s_x;\theta) = \sum_i \frac1{Z_i}\sum_{j\neq i} e^{\frac{-|t_i-t_j|}{\tau}} logP(y_j |x,y_i;\theta)$

where$Zi = \sum_{j\neq i} e^{\frac{-|t_i-t_j|}{\tau}} $ is a normalizing constant, and τ is a hyperparameter.If τ is small, weights will
concentrate on the temporally close pairs. Conversely, the larger the τ is, the smoother the weights are. In this case, more long-term effects will be considered.

$logP(y_j =k|x,y_i;\theta) = \frac{exp(U_X[k,:]^T v_x + U_Y[k,:]^T v_{yi} + b)}{\sum_l exp(U_X[l,:]^T v_x + U_Y[l,:]^T v_{yi} + b)}$

$v_x, v_{yi}$ 是x和yi的embeddings，计算下列式子代替前面的式子

$logP(s_x;\theta) = \sum_i \frac1{N(i)}\sum_{j\in N(i)} logP(y_j |x,y_i;\theta)$

where N(i) = {i1, . . . , ic } is the “context” ofyi . c is a pre-chosen hyper-parameter indicating the length of context, and ik is selected randomly with the probability proportional to $e^{\frac{-|t_i-t_j|}{\tau}}$. N(i) is a multiset, which means it allows duplicates of elements.

**Embedding Tensors**

在不同的场景下有不同的embeddings，例如投资者在买卖股票时有不同的策略。所以结点的embeddings是一个tensor，$\mathcal{T} \in \mathfrak{R}^{V\times K \times D}$ , where V is the size of nodes set and D corresponds to the number of tensor slices.

Given a tuple of attributes a = (a1, . . . , am), and an attributes encoder f , we can get an attribute vector $d =f (a) \in \mathfrak{R}^D$. Then we can compute attributed-gated embeddings as $E^d = \sum_{i=1}^D d_i \mathcal{T}[:,:,i]$ .

However, fully 3-way tensors are not practical because of enormous size. It is a common way to factor the tensor by introducing three matrices $ W^{fv} \in \mathfrak{R}^{F\times V}, W^{fd} \in \mathfrak{R}^{F\times D}, W^{fk} \in \mathfrak{R}^{F\times K}$, and re-reprent $E^d$ by the equation $E^d = (W^{fv})^T · diag(W^{fd}d) · W^{fk}$

where diag(·) denotes the matrix with its argument on the diagonal. These matrices are parametrized by a pre-chosen number of factors F.  It can be seen as the embeddings conditioned on d, and we let
$E^d = (W^{fv})^T W^{fk}$ denote unconditional embeddings.

**IGE: A Multiplicative Neural Model**

用$P(y_j |x,y_i, a_i, a_j;\theta)$代替上面式子中的 $P(y_j |x,y_i;\theta)$

$logP(y_j =k|x,y_i, a_i, a_j;\theta) = \frac{exp(U_X[k,:]^T v_x + U_Y^{d_j}[k,:]^T v_{yi}^{d_i} + b)}{\sum_l exp(U_X[l,:]^T v_x + U_Y^{d_j}[l,:]^T v_{yi}^{d_j} + b)}$

交替训练：

1.选择一个x, 选择$s_x$的两条边，根据$logP(y_j |x,y_i, a_i, a_j;\theta^{(t-1)})$ 计算出$\Delta \theta$, 更新$\theta^{(t)}=\theta^{(t-1)}+\alpha\lambda\Delta\theta$

2.选择一个y, 选择$s_y$的两条边，根据$logP(x_t |y,x_s, a_s, a_t;\theta^{(t-1)})$ 计算出$\Delta \theta$, 更新$\theta^{(t)}=\theta^{(t-1)}+(1-\alpha)\lambda\Delta\theta$



Zhang Y, Xiong Y, Kong X, et al. 2017

Learning Node Embeddings in Interaction Graphs

[C]//Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. ACM