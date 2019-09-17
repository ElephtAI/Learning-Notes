# Summary for the Graph & Network relevant readings

(1) `Jul 2018` [Automatic Opioid User Detection From Twitter: Transductive Ensemble Built On Different Meta-graph Based Similarities Over Heterogeneous Information Network](https://www.ijcai.org/proceedings/2018/466) *[Yujie Fan, Yiming Zhang, Yanfang Ye, Xin Li]*

* **Abstract**: 

> Opioid (e.g., heroin and morphine) addiction has become one of the largest and deadliest epidemics in the United States. To combat such deadly epidemic, in this paper, we propose a novel framework named **HinOPU** to automatically detect opioid users from Twitter, which will assist in sharpening our understanding toward the behavioral process of opioid addiction and treatment. In HinOPU, to model the users and the posted tweets as well as their rich relationships, we introduce **structured heterogeneous information network** (HIN) for representation. Afterwards, we use **meta-graph** based approach to characterize the semantic relatedness over users; we then formulate different similarities over users based on different meta-graphs on HIN. To reduce the cost of acquiring labeled samples for supervised learning, we propose a transductive classification method to build the base classifiers based on different similarities formulated by different meta-graphs. Then, to further improve the detection accuracy, we construct an **ensemble** to combine different predictions from different base classifiers for opioid user detection. Comprehensive experiments on real sample collections from Twitter are conducted to validate the effectiveness of HinOPU in opioid user detection by comparisons with other alternate methods.

* **Key notes**: 
    - **Structured heterogenoeous information network** (*representation*)
    - **meta-graph** (*characterize semantic relatedness over users, formulate similarities*)
    - **Transductive classification** (*semi-supervised, reduce the cost of acquiring labeled samples, vs inductive models*)
    - **Ensemble method**

---

(2) `Oct 2017` [Graph Attention Network](https://arxiv.org/abs/1710.10903) *[Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio]*

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

(3) `Nov 2015` [GATED GRAPH SEQUENCE NEURAL NETWORKS](https://arxiv.org/abs/1511.05493) *[Yujia Li, Daniel Tarlow, Marc Brockschmidt, Richard Zemel]*

* **Abstract**: 

> Graph-structured data appears frequently in domains including chemistry, natural language semantics, social networks, and knowledge bases. In this work, we study **feature learning** techniques for graph-structured inputs. Our starting point is previous work on Graph Neural Networks (Scarselli et al., 2009), which we modify to use gated recurrent units and modern optimization techniques and then extend to output sequences. The result is a flexible and broadly useful class of neural network models that has favorable inductive biases relative to purely sequence-based models (e.g., LSTMs) when the problem is graph-structured. We demonstrate the capabilities on some simple AI (bAbI) and graph algorithm learning tasks. We then show it achieves state-of-the-art performance on a problem from program verification, in which subgraphs need to be matched to abstract data structures.

---

(4) `Sep 2016` [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) *[Thomas N. Kipf, Max Welling]*

* **Abstract**: 

> We present a scalable approach for **semi-supervised learning** on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs. We motivate the choice of our convolutional architecture via a localized first-order approximation of spectral graph convolutions. Our model scales linearly in the number of graph edges and learns hidden layer representations that encode both local grapsh structure and features of nodes. In a number of experiments on citation networks and on a knowledge graph dataset we demonstrate that our approach outperforms related methods by a significant margin.


---

(5) `Oct 2017` [Watch Your Step: Learning Node Embeddings via Graph Attention](https://arxiv.org/abs/1710.09599) *[Sami Abu-El-Haija, Bryan Perozzi, Rami Al-Rfou, Alex Alemi]*

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

(6) `Jun 2017` [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) *[William L. Hamilton, Rex Ying, Jure Leskovec]* `MoNet`

* **Abstract**: 

> **Low-dimensional** embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks, from content recommendation to identifying protein functions. However, most existing approaches require that all nodes in the graph are present during training of the embeddings; these previous approaches are <u>inherently transductive</u> and do not naturally generalize to unseen nodes. Here we present GraphSAGE, a general, **inductive framework** that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. <u>Instead of training individual embeddings for each node, we learn a function that generates embeddings by **sampling and aggregating features** from a node's local neighborhood</u>. Our algorithm outperforms strong baselines on three inductive node-classification benchmarks: we classify the category of unseen nodes in evolving information graphs based on citation and Reddit post data, and we show that our algorithm generalizes to completely unseen graphs using a multi-graph dataset of protein-protein interactions.

---

(7) [Network Lasso: Clustering and Optimization in Large Graphs](https://arxiv.org/abs/1507.00280)

* **Abstract**: 

> Convex optimization is an essential tool for modern data analysis, as it provides a framework to formulate and solve many problems in machine learning and data mining. However, general convex optimization solvers do not scale well, and scalable solvers are often specialized to only work on a narrow class of problems. Therefore, there is a need for simple, scalable algorithms that can solve many common optimization problems. In this paper, we introduce the **network lasso**, a generalization of the group lasso to a network setting that allows for simultaneous clustering and optimization on graphs. We develop an algorithm based on the **Alternating Direction Method of Multipliers (ADMM)** to solve this problem in a distributed and scalable manner, which allows for guaranteed global convergence even on large graphs. We also examine a non-convex extension of this approach. We then demonstrate that many types of problems can be expressed in our framework. We focus on three in particular - binary classification, predicting housing prices, and event detection in time series data - comparing the network lasso to baseline approaches and showing that it is both a fast and accurate method of solving large optimization problems.

---

(8) [Semi-Supervised Classification on Non-Sparse Graphs Using Low-Rank Graph Convolutional Networks](https://arxiv.org/abs/1905.10224)*[Dominik Alfke, Martin Stoll]*

* **Abstract**: 

> Graph Convolutional Networks (GCNs) have proven to be successful tools for semi-supervised learning on graph-based datasets. For sparse graphs, linear and polynomial filter functions have yielded impressive results. For large non-sparse graphs, however, network training and evaluation becomes prohibitively expensive. By introducing low-rank filters, we gain significant runtime acceleration and simultaneously improved accuracy. We further propose an architecture change mimicking techniques from Model Order Reduction in what we call a reduced-order GCN. Moreover, we present how our method can also be applied to hypergraph datasets and how hypergraph convolution can be implemented efficiently.

* **Key notes**: 
    - 

---

(9) [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)*[William L. Hamilton, Rex Ying, Jure Leskovec]* `Negative sampling`

* **Abstract**: 

> Low-dimensional embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks, from content recommendation to identifying protein functions. However, most existing approaches require that all nodes in the graph are present during training of the embeddings; these previous approaches are inherently transductive and do not naturally generalize to unseen nodes. Here we present GraphSAGE, a general, inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, we learn a function that generates embeddings by sampling and aggregating features from a node's local neighborhood. Our algorithm outperforms strong baselines on three inductive node-classification benchmarks: we classify the category of unseen nodes in evolving information graphs based on citation and Reddit post data, and we show that our algorithm generalizes to completely unseen graphs using a multi-graph dataset of protein-protein interactions.

* **Key notes**: 
    - 

---

(10) [Learning Edge Representations via Low-Rank Asymmetric Projections](https://arxiv.org/abs/1705.05615)*[Sami Abu-El-Haija, Bryan Perozzi, Rami Al-Rfou]* `Graph likelihood`

* **Abstract**: 

> We propose a new method for embedding graphs while preserving directed edge information. Learning such continuous-space vector representations (or embeddings) of nodes in a graph is an important first step for using network information (from social networks, user-item graphs, knowledge bases, etc.) in many machine learning tasks.
Unlike previous work, we (1) explicitly model an edge as a function of node embeddings, and we (2) propose a novel objective, the "graph likelihood", which contrasts information from sampled random walks with non-existent edges. Individually, both of these contributions improve the learned representations, especially when there are memory constraints on the total size of the embeddings. When combined, our contributions enable us to significantly improve the state-of-the-art by learning more concise representations that better preserve the graph structure.
We evaluate our method on a variety of link-prediction task including social networks, collaboration networks, and protein interactions, showing that our proposed method learn representations with error reductions of up to 76% and 55%, on directed and undirected graphs. In addition, we show that the representations learned by our method are quite space efficient, producing embeddings which have higher structure-preserving accuracy but are 10 times smaller.

* **Key notes**: 
    - 

---

(11) [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)*[Bryan Perozzi, Rami Al-Rfou, Steven Skiena]* `DeepWalk` 

* **Abstract**: 

> We present DeepWalk, a novel approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs. DeepWalk uses local information obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences. We demonstrate DeepWalk's latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, Flickr, and YouTube. Our results show that DeepWalk outperforms challenging baselines which are allowed a global view of the network, especially in the presence of missing information. DeepWalk's representations can provide F1 scores up to 10% higher than competing methods when labeled data is sparse. In some experiments, DeepWalk's representations are able to outperform all baseline methods while using 60% less training data. DeepWalk is also scalable. It is an online learning algorithm which builds useful incremental results, and is trivially parallelizable. These qualities make it suitable for a broad class of real world applications such as network classification, and anomaly detection.

* **Key notes**: 
    - 

---

(12) [A Tutorial on Network Embeddings](https://arxiv.org/abs/1808.02590)*[Haochen Chen, Bryan Perozzi, Rami Al-Rfou, Steven Skiena]*

* **Abstract**: 

> Network embedding methods aim at learning low-dimensional latent representation of nodes in a network. These representations can be used as features for a wide range of tasks on graphs such as classification, clustering, link prediction, and visualization. In this survey, we give an overview of network embeddings by summarizing and categorizing recent advancements in this research field. We first discuss the desirable properties of network embeddings and briefly introduce the history of network embedding algorithms. Then, we discuss network embedding methods under different scenarios, such as supervised versus unsupervised learning, learning embeddings for homogeneous networks versus for heterogeneous networks, etc. We further demonstrate the applications of network embeddings, and conclude the survey with future work in this area.

* **Key notes**: 
    - 

---

(13) [Link Prediction Based on Graph Neural Networks](https://arxiv.org/abs/1802.09691)*[Muhan Zhang, Yixin Chen]*

* **Abstract**: 

> Link prediction is a key problem for network-structured data. Link prediction heuristics use some score functions, such as common neighbors and Katz index, to measure the likelihood of links. They have obtained wide practical uses due to their simplicity, interpretability, and for some of them, scalability. However, every heuristic has a strong assumption on when two nodes are likely to link, which limits their effectiveness on networks where these assumptions fail. In this regard, a more reasonable way should be learning a suitable heuristic from a given network instead of using predefined ones. By extracting a local subgraph around each target link, we aim to learn a function mapping the subgraph patterns to link existence, thus automatically learning a heuristic' that suits the current network. In this paper, we study this heuristic learning paradigm for link prediction. First, we develop a novel γ-decaying heuristic theory. The theory unifies a wide range of heuristics in a single framework, and proves that all these heuristics can be well approximated from local subgraphs. Our results show that local subgraphs reserve rich information related to link existence. Second, based on the γ-decaying theory, we propose a new algorithm to learn heuristics from local subgraphs using a graph neural network (GNN). Its experimental results show unprecedented performance, working consistently well on a wide range of problems.

* **Key notes**: 
    - 

---

(14) [Deep Graph Infomax](https://arxiv.org/abs/1809.10341)*[Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, R Devon Hjelm]*

* **Abstract**: 

> We present Deep Graph Infomax (DGI), a general approach for learning node representations within graph-structured data in an unsupervised manner. DGI relies on maximizing mutual information between patch representations and corresponding high-level summaries of graphs---both derived using established graph convolutional network architectures. The learnt patch representations summarize subgraphs centered around nodes of interest, and can thus be reused for downstream node-wise learning tasks. In contrast to most prior approaches to unsupervised learning with GCNs, DGI does not rely on random walk objectives, and is readily applicable to both transductive and inductive learning setups. We demonstrate competitive performance on a variety of node classification benchmarks, which at times even exceeds the performance of supervised learning.

* **Key notes**: 
    - 

---

(15) [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)*[Thomas N. Kipf, Max Welling]*

* **Abstract**: 

> We introduce the variational graph auto-encoder (VGAE), a framework for unsupervised learning on graph-structured data based on the variational auto-encoder (VAE). This model makes use of latent variables and is capable of learning interpretable latent representations for undirected graphs. We demonstrate this model using a graph convolutional network (GCN) encoder and a simple inner product decoder. Our model achieves competitive results on a link prediction task in citation networks. In contrast to most existing models for unsupervised learning on graph-structured data and link prediction, our model can naturally incorporate node features, which significantly improves predictive performance on a number of benchmark datasets.

* **Key notes**: 
    - 

---

(16) [DeepGCNs: Can GCNs Go as Deep as CNNs?](https://arxiv.org/abs/1904.03751)*[Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem]*

* **Abstract**: 

> Convolutional Neural Networks (CNNs) achieve impressive performance in a wide variety of fields. Their success benefited from a massive boost when very deep CNN models were able to be reliably trained. Despite their merits, CNNs fail to properly address problems with non-Euclidean data. To overcome this challenge, Graph Convolutional Networks (GCNs) build graphs to represent non-Euclidean data, borrow concepts from CNNs, and apply them in training. GCNs show promising results, but they are usually limited to very shallow models due to the vanishing gradient problem. As a result, most state-of-the-art GCN models are no deeper than 3 or 4 layers. In this work, we present new ways to successfully train very deep GCNs. We do this by borrowing concepts from CNNs, specifically residual/dense connections and dilated convolutions, and adapting them to GCN architectures. Extensive experiments show the positive effect of these deep GCN frameworks. Finally, we use these new concepts to build a very deep 56-layer GCN, and show how it significantly boosts performance (+3.7% mIoU over state-of-the-art) in the task of point cloud semantic segmentation. We believe that the community can greatly benefit from this work, as it opens up many opportunities for advancing GCN-based research.

* **Key notes**: 
    - 

---

(17) [vGraph: A Generative Model for Joint Community Detection and Node Representation Learning](https://arxiv.org/abs/1906.07159)*[Fan-Yun Sun, Meng Qu, Jordan Hoffmann, Chin-Wei Huang, Jian Tang]*

* **Abstract**: 

> This paper focuses on two fundamental tasks of graph analysis: community detection and node representation learning, which capture the global and local structures of graphs, respectively. In the current literature, these two tasks are usually independently studied while they are actually highly correlated. We propose a probabilistic generative model called vGraph to learn community membership and node representation collaboratively. Specifically, we assume that each node can be represented as a mixture of communities, and each community is defined as a multinomial distribution over nodes. Both the mixing coefficients and the community distribution are parameterized by the low-dimensional representations of the nodes and communities. We designed an effective variational inference algorithm which regularizes the community membership of neighboring nodes to be similar in the latent space. Experimental results on multiple real-world graphs show that vGraph is very effective in both community detection and node representation learning, outperforming many competitive baselines in both tasks. We show that the framework of vGraph is quite flexible and can be easily extended to detect hierarchical communities.

* **Key notes**: 
    - 

---

(18) [Bayesian graph convolutional neural networks for semi-supervised classification](https://arxiv.org/abs/1811.11103)*[Yingxue Zhang, Soumyasundar Pal, Mark Coates, Deniz Üstebay]*

* **Abstract**: 

> Recently, techniques for applying convolutional neural networks to graph-structured data have emerged. Graph convolutional neural networks (GCNNs) have been used to address node and graph classification and matrix completion. Although the performance has been impressive, the current implementations have limited capability to incorporate uncertainty in the graph structure. Almost all GCNNs process a graph as though it is a ground-truth depiction of the relationship between nodes, but often the graphs employed in applications are themselves derived from noisy data or modelling assumptions. Spurious edges may be included; other edges may be missing between nodes that have very strong relationships. In this paper we adopt a Bayesian approach, viewing the observed graph as a realization from a parametric family of random graphs. We then target inference of the joint posterior of the random graph parameters and the node (or graph) labels. We present the Bayesian GCNN framework and develop an iterative learning procedure for the case of assortative mixed-membership stochastic block models. We present the results of experiments that demonstrate that the Bayesian formulation can provide better performance when there are very few labels available during the training process.

* **Key notes**: 
    - 

---

(19) [Supervised Community Detection with Line Graph Neural Networks](https://arxiv.org/abs/1705.08415)*[Zhengdao Chen, Xiang Li, Joan Bruna]*

* **Abstract**: 

> We study data-driven methods for community detection on graphs, an inverse problem that is typically solved in terms of the spectrum of certain operators or via posterior inference under certain probabilistic graphical models. Focusing on random graph families such as the stochastic block model, recent research has unified both approaches and identified both statistical and computational signal-to-noise detection thresholds. This graph inference task can be recast as a node-wise graph classification problem, and, as such, computational detection thresholds can be translated in terms of learning within appropriate models. We present a novel family of Graph Neural Networks (GNNs) and show that they can reach those detection thresholds in a purely data-driven manner without access to the underlying generative models, and even improve upon current computational thresholds in hard regimes. For that purpose, we propose to augment GNNs with the non-backtracking operator, defined on the line graph of edge adjacencies. We also perform the first analysis of optimization landscape on using GNNs to solve community detection problems, demonstrating that under certain simplifications and assumptions, the loss value at the local minima is close to the loss value at the global minimum/minima. Finally, the resulting model is also tested on real datasets, performing significantly better than previous models.

* **Key notes**: 
    - 

---

(20) [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997)*[Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann]*

* **Abstract**: 

> Neural message passing algorithms for semi-supervised classification on graphs have recently achieved great success. However, for classifying a node these methods only consider nodes that are a few propagation steps away and the size of this utilized neighborhood is hard to extend. In this paper, we use the relationship between graph convolutional networks (GCN) and PageRank to derive an improved propagation scheme based on personalized PageRank. We utilize this propagation procedure to construct a simple model, personalized propagation of neural predictions (PPNP), and its fast approximation, APPNP. Our model's training time is on par or faster and its number of parameters on par or lower than previous models. It leverages a large, adjustable neighborhood for classification and can be easily combined with any neural network. We show that this model outperforms several recently proposed methods for semi-supervised classification in the most thorough study done so far for GCN-like models. Our implementation is available online.

* **Key notes**: 
    - 

---


() [Template]()*[]*

* **Abstract**: 

> 

* **Key notes**: 
    - 

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

