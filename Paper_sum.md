# Readings summarization for the Graph & Network relevant

(1) [Automatic Opioid User Detection From Twitter: Transductive Ensemble Built On Different Meta-graph Based Similarities Over Heterogeneous Information Network](https://www.ijcai.org/proceedings/2018/466)

* **Abstract**: 

> Opioid (e.g., heroin and morphine) addiction has become one of the largest and deadliest epidemics in the United States. To combat such deadly epidemic, in this paper, we propose a novel framework named **HinOPU** to automatically detect opioid users from Twitter, which will assist in sharpening our understanding toward the behavioral process of opioid addiction and treatment. In HinOPU, to model the users and the posted tweets as well as their rich relationships, we introduce **structured heterogeneous information network** (HIN) for representation. Afterwards, we use **meta-graph** based approach to characterize the semantic relatedness over users; we then formulate different similarities over users based on different meta-graphs on HIN. To reduce the cost of acquiring labeled samples for supervised learning, we propose a transductive classification method to build the base classifiers based on different similarities formulated by different meta-graphs. Then, to further improve the detection accuracy, we construct an **ensemble** to combine different predictions from different base classifiers for opioid user detection. Comprehensive experiments on real sample collections from Twitter are conducted to validate the effectiveness of HinOPU in opioid user detection by comparisons with other alternate methods.

* **Key notes**: 
    - **Structured heterogenoeous information network** (*representation*)
    - **meta-graph** (*characterize semantic relatedness over users, formulate similarities*)
    - **Transductive classification** (*semi-supervised, reduce the cost of acquiring labeled samples, vs inductive models*)
    - **Ensemble method**

---

(2) [Graph Attention Network](https://arxiv.org/abs/1710.10903)

* **Abstract**: 

> We present **graph attention networks (GATs)**, novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which <u>nodes are able to attend over their neighborhoods' features</u>, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront. In this way, we address several key challenges of **spectral-based graph neural networks** simultaneously, and make our model readily applicable to inductive as well as transductive problems. Our GAT models have achieved or matched state-of-the-art results across four established transductive and inductive graph benchmarks: the Cora, Citeseer and Pubmed citation network datasets, as well as a protein-protein interaction dataset (wherein test graphs remain unseen during training).

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

> Graph embedding methods represent nodes in a continuous vector space, preserving information from the graph (e.g. by sampling random walks). There are many hyper-parameters to these methods (such as random walk length) which have to be manually tuned for every graph. In this paper, we replace random walk hyper-parameters with trainable parameters that we automatically learn via backpropagation. In particular, we learn a novel attention model on the power series of the transition matrix, which guides the random walk to optimize an upstream objective. Unlike previous approaches to attention models, the method that we propose utilizes attention parameters exclusively on the data (e.g. on the random walk), and not used by the model for inference. We experiment on link prediction tasks, as we aim to produce embeddings that best-preserve the graph structure, generalizing to unseen information. We improve state-of-the-art on a comprehensive suite of real world datasets including social, collaboration, and biological networks. Adding attention to random walks can reduce the error by 20% to 45% on datasets we attempted. Further, our learned attention parameters are different for every graph, and our automatically-found values agree with the optimal choice of hyper-parameter if we manually tune existing methods.

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

### To do

* [ ] Transductive & Inductive Methods from the Graph Attention Network Paper (No.2 above)
* [ ] [Semi-Supervised Classification on Non-Sparse Graphs Using Low-Rank Graph Convolutional Network](https://arxiv.org/abs/1905.10224)
* [ ] W. Hamilton, R. Ying, and J. Leskovec. Inductive representation learning on large graphs. In NIPS, 2017. (*negative sampling*)
* [ ] S. Abu-El-Haija, B. Perozzi, and R. Al-Rfou. Learning edge representations via low-rank asymmetric projections. In ACM International Conference on Information and Knowledge Management (CIKM), 2017. (*graph likelihood*)
* [ ] T. N. Kipf and M. Welling. Variational graph auto-encoders. In NIPS Workshop on Bayesian
Deep Learning, 2016.
* [ ] N.Shervashidze,P.Schweitzer,E.J.v.Leeuwen,K.Mehlhorn,andK.M.Borgwardt.Weisfeiler- lehman graph kernels. Journal of Machine Learning Research, 12:2539–2561, 2011. (*Graph Kernel*)
* [ ] B. Perozzi, R. Al-Rfou, and S. Skiena. Deepwalk: Online learning of social representations. In KDD, 2014. (Deep Walk)

**Knowledge Graph Related**:

* [ ] [Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs](https://www.aclweb.org/anthology/P19-1466)

