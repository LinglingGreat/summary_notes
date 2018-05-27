##Survey 1

**Graph Embedding Problem Settings**

- Graph Embedding Input
  - Homogeneous Graph
    - Undirected，unweighted，directed，weighted
    - Challenge: How to capture the diversity of connectivity patterns observed in graphs
  - Heterogeneous Graph
    - Community-based question answering (cQA) sites，Multimedia Networks，Knowledge Graphs
    - Challenge: How to explore global consistency between different types of objects, and how to deal with the imbalances of different-typed objects, if any
  - Graph with Auxiliary Information
    - Label，Attribute，Node feature，Information propagation，Knowledge base
    - Challenge: How to incorporate the rich and unstructured information so that the learnt embeddings are both representing the topological structure and discriminative in terms of the auxiliary information.
  - Graph Constructed from Non-relational Data
    - Challenge: How to construct a graph that encodes the pairwise relations between instances and how to preserve the generated node proximity matrix in the embedded space.
- Graph Embedding Output：
- how to find a suitable type of embedding output which meets the needs of the specific application task
  - Node Embedding
    - Challenge: How to define the node pairwise proximity in various types of input graph and how to encode the proximity in the learnt embeddings.
  - Edge Embedding
    - Challenge: How to define the edge-level similarity and how to model the asymmetric property of the edges, if any.
  - Hybrid Embedding
    - Substructure embedding，community embedding
    - Challenge: How to generate the target substructure and how to embed different types of graph components in one common space.
  - Whole-Graph Embedding
    - Challenge: How to capture the properties of a whole graph and how to make a trade-off between expressivity and efficiency.

**Graph Embedding Techniques**

- Matrix Factorization
  - Graph Laplacian Eigenmaps
    - The graph property to be preserved can be interpreted as pairwise node similarities, thus a larger penalty is imposed if two nodes with larger similarity are embedded far apart
  - Node Proximity Matrix Factorization
    - Node proximity can be approximated in low-dimensional space using matrix factorization. The objective of preserving the node proximity is to minimize the loss during the approximation
- Deep Learning
  - With Random Walk
    - The second-order proximity in a graph can be preserved in the embedded space by maximizing the probability of observing the neighbourhood of a node conditioned on its embedding
    - There are usually two solutions to approximate the full softmax: hierarchical softmax and negative sampling.
  - Without Random Walk
    - The multi-layered learning architecture is a robust and effective solution to encode the graph into a low dimensional space
    - Autoencoder，Deep Neural Network，Others
- Edge Reconstruction
  - Maximize Edge Reconstruct Probability
    - Good node embedding maximizes the probability of generating the observed edges in a graph
  - Minimize Distance-based Loss
    - The node proximity calculated based on node embedding should be as close to the node proximity calculated based on the observed edges as possible
  - Minimize Margin-based Ranking Loss
    - A node’s embedding is more similar to the embedding of relevant nodes than that of any other irrelevant node
- Graph Kernel
- The whole graph structure can be represented as a vector containing the counts of elementary substructures that are decomposed from it
  - Based on Graphlet
  - Based on Subtree Patterns
  - Based on Random Walks
- Generative Model
  - Embed Graph into Latent Space
    - Nodes are embedded into the latent semantic space where the distances among node embeddings can explain the observed graph structure
  - Incorporate Semantics for Embedding
    - Nodes who are not only close in the graph but also having similar semantics should be embedded closer. The node semantics can be detected from node descriptions via a generative model
- Hybrid Techniques and Others

**Applications**

- Node Related Applications
  - Node Classification
  - Node Clustering
  - Node Recommendation/Retrieval/Ranking
- Edge Related Applications
  - Link Prediction and Graph Reconstruction
  - Triple Classification
- Graph Related Applications
  - Graph Classification
  - Visualization
- Other Applications
  - Knowledge graph related
  - Multimedia network related
  - Information propagation related
  - Social networks alignment



Cai H, Zheng V W, Chang K. 

A comprehensive survey of graph embedding: problems, techniques and applications

[J]. IEEE Transactions on Knowledge and Data Engineering, 2018. 



## Survey 2

**一、Embedding nodes**

1. **Overview of approaches: An encoder-decoder perspective**

The intuition behind the encoder-decoder idea is the following: if we can learn to decode high-dimensional graph information—such as the global positions of nodes in the graph or the structure of local graph neighborhoods—from encoded low-dimensional embeddings, then, in principle, these embeddings should contain all information necessary for downstream machine learning tasks

encoder是将结点转换成embeddings，decoder是接受一系列embeddings，然后从中解码出用户指定的统计量，比如结点属于哪个类，两个结点之间是否存在边等。

目标是优化encoder和decoder的mapping来最小化误差或损失。损失是decoder的结果和真实结果之间的差异。

四个方面的不同：pairwise similarityfunction, encoder function, decoder function, loss function

2. **Shallow embedding approaches**

For these shallow embedding approaches, the encoder function—which maps nodes to vector embeddings—is simply an “embedding lookup”

- Factorization-based approaches
  - Laplacian eigenmaps：decoder是两个向量之差的L2范数的平方，损失函数是decoder的结果的加权和，权重是两个结点的相似性。
  - Inner-product methods：decoder是两个向量的内积，例如The Graph Factorization (GF) algorithm, GraRep, and HOPE，他们的损失函数都是MSE：decoder的结果和实际相似度的差的L2范数的平方。他们的不同之处是相似性的度量。GF直接用邻接矩阵，GraRep用邻接矩阵的平方，HOPE用based on
    Jaccard neighborhood overlaps
  - 他们的共同点是：Loss function基本上是：$||Z^TZ-S||_2^2$ ,  embedding矩阵Z和相似性矩阵S
- Random walk approaches
  - DeepWalk and node2vec：decoder是从结点i出发在T步内经过结点j的概率；交叉熵损失函数
  - Large-scale information network embeddings (LINE)
  - HARP: Extending random-walk embeddings via graph pre-processing：           a graph coarsening procedure is used to collapse related nodes in G together into “supernodes”, and then DeepWalk, node2vec, or LINE is run on this coarsened graph. After embedding the coarsened version of G, the learned embedding of each supernode is used as an initial value for the random walk embeddings of the supernode’s constituent nodes
  - Additional variants of the random-walk idea

缺点：

1.No parameters are shared between nodes in the encoder

2.Shallow embedding also fails to leverage node attributes during encoding

3.Shallow embedding methods are inherently transductive



3. **Generalized encoder-decoder architectures**

- Neighborhood autoencoder methods：       they use autoencoders，例如DNGR, SDNE, Extract high-dimensional neighborhood vector,(si contains vi’s proximity to all other nodes),Compress si to low-dimensional embedding
- Neighborhood aggregation and convolutional encoders：they generate embeddings for a node by aggregating information from its local neighborhood，例如GCN，column networks，GraphSAGE



4. **Incorporating task-specific supervision**

cross-entropy loss,  backpropagation

5. **Extensions to multi-modal graphs**

- Dealing with different node and edge types
- Tying node embeddings across layers：如OhmNet



6. **Embedding structural roles**

struc2vec，GraphWave

7. **Applications of node embeddings**

- Visualization and pattern discovery
- Clustering and community detection
- Node classification and semi-supervised learning
- Link prediction



**二、Embedding subgraphs**

1. **Sets of node embeddings and convolutional approaches**

The basic intuition behind these approaches is that they equate subgraphs with sets of node embeddings

- Sum-based approaches
- Graph-coarsening approaches
- Further variations

2. **Graph neural networks**

GNN，MPNNs

3. **Applications of subgraph embeddings**



Hamilton W L, Ying R, Leskovec J. 

Representation Learning on Graphs: Methods and Applications

[J]. arXiv preprint arXiv:1709.05584, 2017. 

