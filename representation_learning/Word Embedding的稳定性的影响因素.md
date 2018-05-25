We define stability as the percent overlap between nearest neighbors in an embedding space

两个embedding空间，找到同一个词的最近的十个邻居，将两个邻居列表的重叠作为词W的stability。

多个embedding空间时，considering the average overlap between two sets of embedding spaces

建立回归模型：

自变量是(1) properties related to the word itself; (2) properties of the data used to train the embeddings; and (3) properties of the algorithm used to construct these embeddings.

岭回归，最小化下列函数：

$L(w) = \frac12 \sum_{n=1}^N (y_n - w^T x_n)^2 + \frac{\lambda}2 ||w||^2$

we set 正则化常数 $\lambda = 1$

**Word Properties：**

Primary part-of-speech： Adjective
Secondary part-of-speech： Noun

Parts-of-speech： 2

WordNet senses： 3

Syllables： 5		音节

**Data Properties**

Raw frequency in corpus A： 106
Raw frequency in corpus B： 669
Diff. in raw frequency： 563
Vocab. size of corpus A： 10,508
Vocab. size of corpus B： 43,888
Diff. in vocab. size： 33,380
Overlap in corpora vocab.： 17%
Domains present： Arts, Europarl
Do the domains match?： False
Training position in A： 1.02%
Training position in B： 0.15%
Diff. in training position： 0.86%
**Algorithm Properties**
Algorithms present： word2vec, PPMI
Do the algorithms match?： False
Embedding dimension of A： 100
Embedding dimension of B： 100
Diff. in dimension： 0
Do the dimensions match?：True



we show that domain and part-of-speech are key factors of instability

In order to use the most stable embedding spaces for future tasks, we recommend either
using GloVe or learning a good curriculum for word2vec training data. We also recommend
using in-domain embeddings whenever possible。



Wendlandt L, Kummerfeld J K, Mihalcea R. 

Factors Influencing the Surprising Instability of Word Embeddings

[J]. arXiv preprint arXiv:1804.09692, 2018. 