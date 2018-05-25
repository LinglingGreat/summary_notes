Sac2Vec(structure and content to vector), a network representation technique using structure and content. It is a multi-layered graph approach which uses a random walk to generate the node embedding.

G = (V, E, W, F), F can be considered as the content matrix.其中$f_i$ 是the word vector (content) associated with the node

F可以用词袋模型(bag-of-words)表示，矩阵的每一行是相应结点的文本内容的tf-idf向量，所以F的维度是n×d，d是语料中的unique words的数量。(预处理之后)

给定输入网络G=(V,E,W,F),分为两层：

Structure Layer:  这一层是$G_s=(V,E_s,W_s) , E_s=E, W_s=W$

Content Layer:  这一层是有向图$G_c=(V,E_c,W_c) , w_{ij}^c=Cosine(F_i, F_j), 但是只保留top \; \theta \times \lceil avg_s \rceil的出去的边，其中\theta是一个正整数，avg_s $ 是Structure layer的结点的平均出度，|E|/n(有向)，2×|E|/n(无向)

**Convex Sum of Embeddings: CSoE**

用node2vec为结构层和内容层独立地生成embeddings,然后取两个embeddings的凸组合(维度一致时才能用)

$e_{convex}^i = \alpha e_s^i + (1-\alpha) e_c^i，\forall i \in \lbrace 1,2,...,n \rbrace \; and \; 0 \leq \alpha \leq 1$

**Appended Embeddings: AE**

将两个向量拼接起来,维度不一样也可以

$e_{appended}^i = [e_s^i || e_c^i]，e_s^i \in R^{K_s}, e_c^i \in R^{K_c}, e_{appended}^i \in R^{K_s+K_c}, \forall i \in \lbrace 1,2,..,n \rbrace$

**SaC2Vec Model**

为structure layer的结点$v_i^s$定义$\Gamma_i^s$如下：content layer的$\Gamma_i^c$类似

$\Gamma_i^s = \lbrace (v_i^s, v_j^s) \in E_s | w_{v_i^s, v_j^s}^s \geq \frac1{|E_s|}\sum_{e' \in E_s} w_{e'}^s, v_j^s \in V \rbrace$

是i的那些权重大于该层平均边权重的出边的集合。

两层相同结点之间的权重：

$w_i^{sc} = ln(e+|\Gamma_i^s|), w_i^{cs} = ln(e+|\Gamma_i^c|)$

**random walk**

当前所在结点是$v_i$，要么是结构层，要么是内容层，下一步走到哪一层呢？我们的目标是to move to a layer which is more informative in some sense at node $v_i$.

$p(v_i^s|v_i) = \frac{w_i^{cs}}{w_i^{sc}+w_i^{cs}}$

$p(v_i^c|v_i) = 1- p(v_i^s|v_i)=\frac{w_i^{sc}}{w_i^{sc}+w_i^{cs}}$

考虑第一个式子，$w_i^{sc}$越大，结点$v_i^s$ 的那些权重大于结构层的相对高的权重的出边越多，此时random walk如果在结构层，在走下一步时会有很多选择。如果$w_i^{sc}$很小，random walk 处在结构层的话，下一步选择会很少，此时的选择会更有信息丰富性，并且less random.所以，当$w_i^{sc}$很大时，倾向于选择content layer, 当$w_i^{cs}$很大时，倾向于选择structure layer.一旦选择了某一层，在走下一步时就不用考虑另一层了。



**Algorithm 1 SaC2Vec** - Structure and Content to Vector
1:    Input: The network G = (V,E,W, F), K: Dimension of the embedding space where K << min(n, d)(d是distinct words的数量), r: Number of time to start random walk from each vertex, l: Length of each random walk
2:    Output: The node embeddings of the network G
3:    Generate the structure layer and the content layer
4:    Add the edges between the layers with weights to generate the multi-layered network
5:    Corpus = [ ] .                            Initialize the corpus
6:    for iter $\in$ {1, 2, ..., r} do
7:    		for $v \in V$ do
8:   			 select the node v as the starting node of the random walk
9:    			Walk = [v] . 			Initialize the Walk(sequence of nodes)
10:  			for walkIter $\in$ {1, 2, ..., l} do
11: 	 			Select the layer to move next with probabilities
12:  				Move 1 step using node2vec to find the next node $v_i$
13:  				Append $v_i$ to Walk
14:  			end for
15:  			Append Walk to Corpus
16:  		end for
17:  end for
18:  Find the node embeddings by running language model on Corpus(SkipGram模型，最大化给定该结点的向量表示出现context nodes的概率)

实验：node classication, node clustering and network visualization

It means SaC2Vec is able to understand the bad quality of the content layer during the learning process and embeddings were learnt mostly from the structure layer.







Sambaran Bandyopadhyay, Harsh Kara, Anirban Biswas, M N Murty，2018.4.27

SaC2Vec: Information Network Representation with Structure and Content

