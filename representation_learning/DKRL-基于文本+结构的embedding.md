Description-Embodied Knowledge Representation Learning

每个头结点和尾结点有两个向量，分别是基于结构s的向量和基于文本描述d的向量

energy function:

$E = E_S + E_D,   E_D = E_{DD} + E_{DS}+E_{SD}$

$E_{DD}=||h_d+r-t_d||, E_{DS}=||h_d+r-t_s||, E_{SD}=||h_s+r-t_d||$

two encoders to build description-based representations

**Continuous Bag-of-words Encoder**

为每个实体选择文本描述中的top n个关键词(可以用TF-IDF进行排序)作为输入，将关键词的embeddings加起来作为实体的embedding，用来最小化$E_D$

**Convolutional Neural Network Encoder**

5层，实体的预处理后的文本描述作为输入，输出该实体基于文本描述的embedding.

预处理：去停用词，标记文本描述中的短语，将它们作为词，每个词有一个word embedding，作为CNN的输入。









R Xie，Z Liu，J Jia，... - 2016

Representation Learning of Knowledge Graphs with Entity Descriptions

AAAI