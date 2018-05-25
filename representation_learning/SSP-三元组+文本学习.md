semantic space projection (SSP) model which jointly learns from the symbolic triples and textual descriptions

损失函数：

$f_r(h,t) = -\lambda ||e-s^Tes||_2^2 + ||e||_2^2，e=h+r-t$

the component of the loss in the normal vector direction is ($s^Tes$), then the other orthogonal one, that is inside the hyperplane, is ($e-s^Tes$).

$超平面的法向量s=S(s_h, s_t)$

将实体描述看作文档，采用主题模型，得到文档的主题分布作为实体的语义向量s，如(0.1, 0.9, 0.0)表示该实体的主题应该是2

$S(s_h, s_t) = \frac{s_h+s_t}{||s_h+s_t||_2^2}$

Standard setting: 给定预训练好的语义向量，模型中固定它们然后优化其它参数

Joint setting: 同时实施主题模型和embedding模型，此时三元组也会给文本语义带来正向影响。

Total Loss:

$L = L_{embed} + \mu L_{topic}$

$L_{embed} = \sum_{(h,r,t) \in \Delta, (h',r',t') \in \Delta'} [f_{r'}(h', t') - f_r(h, t)+\gamma]_+$

$L_{topic} = \sum_{e \in E, w \in D_e} (C_{e,w}-s_e^Tw)^2$

where E is the set of entities, and $D_e$ is the set of words in the description of entity e. $C_{e,w}$ is the times of the word w occurring in the description e. $s_e$ is the semantic vector of entity e and w is the topic distribution of word w. Similarly, SGD is applied in the optimization.

实验：知识图谱补全，实体预测

Xiao H, Huang M, Meng L, et al. 

SSP: Semantic Space Projection for Knowledge Graph Embedding with Text Descriptions

[C]//AAAI. 2017, 17: 3104-3110. 