# Summary for the Graph & Network relevant readings

(1) [Automatic Opioid User Detection From Twitter: Transductive Ensemble Built On Different Meta-graph Based Similarities Over Heterogeneous Information Network](https://www.ijcai.org/proceedings/2018/466) *[Yujie Fan, Yiming Zhang, Yanfang Ye, Xin Li]*

* **Abstract**: 

> Opioid (e.g., heroin and morphine) addiction has become one of the largest and deadliest epidemics in the United States. To combat such deadly epidemic, in this paper, we propose a novel framework named **HinOPU** to automatically detect opioid users from Twitter, which will assist in sharpening our understanding toward the behavioral process of opioid addiction and treatment. In HinOPU, to model the users and the posted tweets as well as their rich relationships, we introduce **structured heterogeneous information network** (HIN) for representation. Afterwards, we use **meta-graph** based approach to characterize the semantic relatedness over users; we then formulate different similarities over users based on different meta-graphs on HIN. To reduce the cost of acquiring labeled samples for supervised learning, we propose a transductive classification method to build the base classifiers based on different similarities formulated by different meta-graphs. Then, to further improve the detection accuracy, we construct an **ensemble** to combine different predictions from different base classifiers for opioid user detection. Comprehensive experiments on real sample collections from Twitter are conducted to validate the effectiveness of HinOPU in opioid user detection by comparisons with other alternate methods.

* **Key notes**: 
    - **Structured heterogenoeous information network** (*representation*)
    - **meta-graph** (*characterize semantic relatedness over users, formulate similarities*)
    - **Transductive classification** (*semi-supervised, reduce the cost of acquiring labeled samples, vs inductive models*)
    - **Ensemble method**

---

(2) [Graph Attention Network](https://arxiv.org/abs/1710.10903) *[Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio]*

* **Abstract**: 

> We present **graph attention networks (GATs)**, novel neural network architectures that operate on graph-structured data, leveraging masked **self-attentional layers** to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which <u>nodes are able to attend over their neighborhoods' features</u>, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront. In this way, we address several key challenges of **spectral-based graph neural networks** simultaneously, and make our model readily applicable to inductive as well as transductive problems. Our GAT models have achieved or matched state-of-the-art results across four established transductive and inductive graph benchmarks: the Cora, Citeseer and Pubmed citation network datasets, as well as a protein-protein interaction dataset (wherein test graphs remain unseen during training).

* **Key Notes**:
    - Graph Neural Network consists of an **iterative process**: 
        - propagates the node states until equilibrium
        - followed by a neural network, produces an output for each node on its states
    - **Convolution** to the graph domain: 
        - <u>spectral approaches</u>: Computing the **eigendecomposition of the graph Laplacian** in Fourier domain, Chebyshev expansion; **depends on spacial graph structure**
        - <u>non-spectral approches</u>: define convolutions directly on the graph, operating on groups of spatially close neighbors -> [GraphSAGE](https://arxiv.org/abs/1706.02216)
    - **Attention-based** architecture -> **node classification**: *compute the hidden representations of each node in the graph by attending over its **neighbors** following a **self-attention** strategy*
    - Interesting **Properties** of Attention-architecture:
        - **[i]** efficient operation, parallelizable across **node-neighbor pairs**
        - **[ii]** can be applied to graph nodes having different degrees (specifyig arbitrary weights to neighbors)
        - **[iii]** directly applicable to inductive learning problems, the model has to generalize to completely unseen graphs
    - In the attention layer, features from K **independent attention mechanisms** are concatenated to employ **multi-head attention** -> <u>to stabilize the learning process of self-attention</u>
    - The author's attention-layer, works with the **entirety of the neighborhood** and **does not assume any ordering** within it, and it's a **particular instance of** [MoNet](https://arxiv.org/abs/1611.08402)
    - **Future Improvement**: 
        - handle large batch sizes
        - perform a thorough analysis on the model interpretability
        - extending the method to perform **graph calssification**
        - extending the model to incorporate **edge features** (indicating replationship among nodes)

---

(3) [GATED GRAPH SEQUENCE NEURAL NETWORKS](https://arxiv.org/abs/1511å.05493)

* **Abstract**: 

> Graph-structured data appears frequently in domains including chemistry, natural language semantics, social networks, and knowledge bases. In this work, we study **feature learning** techniques for graph-structured inputs. Our starting point is previous work on Graph Neural Networks (Scarselli et al., 2009), which we modify to use gated recurrent units and modern optimization techniques and then extend to output sequences. The result is a flexible and broadly useful class of neural network models that has favorable inductive biases relative to purely sequence-based models (e.g., LSTMs) when the problem is graph-structured. We demonstrate the capabilities on some simple AI (bAbI) and graph algorithm learning tasks. We then show it achieves state-of-the-art performance on a problem from program verification, in which subgraphs need to be matched to abstract data structures.

---

(4) [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

* **Abstract**: 

> We present a scalable approach for **semi-supervised learning** on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs. We motivate the choice of our convolutional architecture via a localized first-order approximation of spectral graph convolutions. Our model scales linearly in the number of graph edges and learns hidden layer representations that encode both local grapsh structure and features of nodes. In a number of experiments on citation networks and on a knowledge graph dataset we demonstrate that our approach outperforms related methods by a significant margin.


---

(5) [Watch Your Step: Learning Node Embeddings via Graph Attention](https://arxiv.org/abs/1710.09599)

* **Abstract**: 

> Graph embedding methods represent nodes in a continuous vector space, preserving information from the graph (e.g. by sampling random walks). There are many hyper-parameters to these methods (such as random walk length) which have to be manually tuned for every graph. In this paper, we replace random walk hyper-parameters with **trainable parameters that we automatically learn via backpropagation**. In particular, we learn a novel attention model on the power series of the transition matrix, which guides the random walk to optimize an upstream objective. Unlike previous approaches to attention models, the method that we propose **utilizes attention parameters exclusively on the data** (e.g. on the random walk), and **ot used by the model for inference**. We experiment on link prediction tasks, as we aim to produce embeddings that best-preserve the graph structure, generalizing to unseen information. We improve state-of-the-art on a comprehensive suite of real world datasets including social, collaboration, and biological networks. **Adding attention to random walks** can reduce the error by 20% to 45% on datasets we attempted. Further, our learned attention parameters are different for every graph, and our automatically-found values agree with the optimal choice of hyper-parameter if we manually tune existing methods.

* **Key notes**: 
    - **Unsupervised graph embedding methods**: 
        - **[i]** sample pair-wise relationships from the graph through **random walks** and **counting node co-occurance**
        - **[ii]** train an embedding model (using *skipgram of word2vec*) to learn representations that encode pairwise node similarities
        - **[Problem]**: significantly depend on hyper-parameters (e.g. length of random walk)
    - In this work, the author replace the hyper-parameter (`C` for *length of random walk*; `Q` for *context distribution*) with **trainable** parameters: (-> automatically learned for each graph)
        - pose **graph embedding** as end2end learning (the above discrete 2 steps random walk co-occurance sampling)
        - followed by **representation learning**: <u>joint using a closed-form expectation over the graph adjacency matrix</u>
    - **contraibution** summary: 
        - **[1]** propose extendible family of graph attention models -> **learn arbitrary context distribution**
        - **[2]** show optimal hyper-parameter (found by manual tunining) agrees
        - **[3]** evaluate a number of challenging link prediction tasks
    - <u>Preliminaries</u>:
        - Graph embedding 
        - **Learning Embeddings via Random Walks**: use w2v on path sequence from random walk
        - Graph Likelihood
    - The authors extend the **Negative Log Graph Likelihood** loss to include **attention parameters** on the random walk sampling: 
        - Expectation on the co-occurance matrix: $E[D]$ to approximate
        - [DeepWalk](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf) do not use $C$ (length of random walk) as a hard limit
        - try to learn the context distribution $Q$ (with *C-dimentional*)
        - train **softmax attentio model** on the **infinite power series** of the transition matrix
    - **Extension**: extend the model to learn the weight of any other type of pair-wise node similarity
    - learn a free-form contexts distribution with a parameter for each type of context similarity (distance in a random walk)

---

(6) [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

* **Abstract**: 

> **Low-dimensional** embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks, from content recommendation to identifying protein functions. However, most existing approaches require that all nodes in the graph are present during training of the embeddings; these previous approaches are <u>inherently transductive</u> and do not naturally generalize to unseen nodes. Here we present GraphSAGE, a general, **inductive framework** that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. <u>Instead of training individual embeddings for each node, we learn a function that generates embeddings by **sampling and aggregating features** from a node's local neighborhood</u>. Our algorithm outperforms strong baselines on three inductive node-classification benchmarks: we classify the category of unseen nodes in evolving information graphs based on citation and Reddit post data, and we show that our algorithm generalizes to completely unseen graphs using a multi-graph dataset of protein-protein interactions.

---

(7) [Network Lasso: Clustering and Optimization in Large Graphs](https://arxiv.org/abs/1507.00280)

* **Abstract**: 

> Convex optimization is an essential tool for modern data analysis, as it provides a framework to formulate and solve many problems in machine learning and data mining. However, general convex optimization solvers do not scale well, and scalable solvers are often specialized to only work on a narrow class of problems. Therefore, there is a need for simple, scalable algorithms that can solve many common optimization problems. In this paper, we introduce the **network lasso**, a generalization of the group lasso to a network setting that allows for simultaneous clustering and optimization on graphs. We develop an algorithm based on the **Alternating Direction Method of Multipliers (ADMM)** to solve this problem in a distributed and scalable manner, which allows for guaranteed global convergence even on large graphs. We also examine a non-convex extension of this approach. We then demonstrate that many types of problems can be expressed in our framework. We focus on three in particular - binary classification, predicting housing prices, and event detection in time series data - comparing the network lasso to baseline approaches and showing that it is both a fast and accurate method of solving large optimization problems.

---

1. [Template](https)

* **Abstract**: 

> template

* **Key notes**: 
    - **Structured heterogenoeous information network** (*representation*)

---

### To do for graph & network

* [ ] [Semi-Supervised Classification on Non-Sparse Graphs Using Low-Rank Graph Convolutional Network](https://arxiv.org/abs/1905.10224)
* [ ] W. Hamilton, R. Ying, and J. Leskovec. Inductive representation learning on large graphs. In NIPS, 2017. (*negative sampling*)
* [ ] S. Abu-El-Haija, B. Perozzi, and R. Al-Rfou. Learning edge representations via low-rank asymmetric projections. In ACM International Conference on Information and Knowledge Management (CIKM), 2017. (*graph likelihood*)
* [ ] N.Shervashidze,P.Schweitzer,E.J.v.Leeuwen,K.Mehlhorn,andK.M.Borgwardt.Weisfeiler- lehman graph kernels. Journal of Machine Learning Research, 12:2539–2561, 2011. (*Graph Kernel*)
* [ ] B. Perozzi, R. Al-Rfou, and S. Skiena. Deepwalk: Online learning of social representations. In KDD, 2014. (Deep Walk)
* [ ] Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodola, Jan Svoboda, and Michael M Bronstein. Geometric deep learning on graphs and manifolds using mixture model cnns. arXiv preprint arXiv:1611.08402, 2016. (*MoNet*)
* [ ] H. Chen, B. Perozzi, R. Al-Rfou, and S. Skiena. A tutorial on network embeddings

**Knowledge Graph Related**:

* [ ] [Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs](https://www.aclweb.org/anthology/P19-1466)

**Other paper**:

* [ ] T. N. Kipf and M. Welling. Variational graph auto-encoders. In NIPS Workshop on Bayesian
Deep Learning, 2016.

