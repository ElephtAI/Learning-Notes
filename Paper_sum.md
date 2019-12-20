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

> Graph-structured data appears frequently in domains including chemistry, natural language semantics, social networks, and knowledge bases. In this work, we study **feature learning** techniques for graph-structured inputs. Our starting point is previous work on Graph Neural Networks (Scarselli et al., 2009), which we modify to use **gated recurrent units** and **modern optimization techniques** and then extend to output sequences. The result is a flexible and broadly useful class of neural network models that has favorable inductive biases relative to purely sequence-based models (e.g., LSTMs) when the problem is graph-structured. We demonstrate the capabilities on some simple AI (bAbI) and graph algorithm learning tasks. We then show it achieves state-of-the-art performance on a problem from program verification, in which subgraphs need to be matched to abstract data structures.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** extension of **Graph Neural Networks** that ouptuts **sequences**
            - as previous only produce single output => cannot be used for problems require outputting sequences <u>e.g. pahts on a graph, enumerations of graph nodes with properties</u>
            - incorporate node labels as additional inputs => node annotations (*easy for the propagation model to learn to propagate the node annotation for s to all nodes reachable*)
        - **[2]** highlighting that GNN are a broadly useful calss of neural network
    - <u>**Other Notes**</u>:
        - Two settings for feature learning on graph: 
            1. learning a representation of the **input graph**
            2. learning representations of the **internal state** during the process of **producing a sequence of outputs**
        - GNN map graphs -> outputs in two steps:
            - **[i]** propagation step computes node representation for each node (*iterative procedure*)
            - **[ii]** output model map from node representations and corresponding lables to output
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:
        - develop end2end methods that take the question as an initial input and then dynamically derive the facts needed to answer the question

---

(4) `Feb 2017` [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) *[Thomas N. Kipf, Max Welling]*

* **Abstract**: 

> We present a **scalable approach** for **semi-supervised learning** on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs. We motivate the choice of our convolutional architecture via a **localized first-order approximation** of spectral graph convolutions. Our model scales linearly in the number of **graph edges** and learns hidden layer representations that encode both local graph structure and features of nodes. In a number of experiments on **citation networks** and on a **knowledge graph dataset** we demonstrate that our approach outperforms related methods by a significant margin.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** Direct encode the graph structure using a <u>neural network model</u> and train on a supervised target -> **avoding expilicit graph-based regularization in the loss function**
        - **[2]** introduce a layer-wise propagation rule for NN models operates directly on graph -> <u>motivated from a first order (linear) approximation of **spectral graph convolutions**</u>
            - use **renormalization trick** to avoid numerical instabilities (gradient exploding/vanishing) while deep learning
        - **[3]** demonstrate such model can be used for **fast and scalable** semi-supervised classification 
            - We can condition the model $f(X,A)$ both on the data $X$ and adjacency matrix $A$ [*especially for scenarios where A contain information nnot present in X*]
        - **[4]** The author's model use a single weigh matrix per layer and deals with varying node degrees through an **appropriate normalization of the adjacency matrix**
        - In this work, the author implicitly assumes: 
            - **locality** (dependence on the kth-order neighborhood for a GCN with K layers)
            - **equal importance** of <u>self-connections</u> vs. <u>edges to neighboring nodes</u>
    - <u>**Other Notes**</u>:
        - In **graph-based semi-supervised learning**: label is smoothed via explicit graph-based **regularization**
        > like using graph Laplacian regularization in the loss fucntion
        - graph representation for semi-supervised learning: 
            - **[i]** use graph Laplacian regularization `label propagation` `manifold regularization` `deep semi-supervised embedding`
            - **[ii]** use graph embedding-based approches
    - <u>**Use cases**</u>:
        - on citation networks
        - on knowledge graph dataset
    - <u>**Further directions**</u>:
        - **[1]** Memory requirement (for very large and densely connected graph datasets, further approximations might be necessary)
        - **[2]** directed edges and edge features (<u>not supported in this work</u>)
        - **[3]** limiting assumptions
            - > **introduce a trade-off parameter $\lambda$** in the definition of $A$ (can be learned by GD)

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

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** propose **GraphSAGE**(sample and aggregate) for inductive node embedding
            - leverate node attributes (e.g. `text attributes`, `node profile information`, `node degrees`) -> generalize to unseen nodes
            - simultaneously learn **topological structure** of each node's neighborhood, and, **distribution** of node features in the neighborhood
            - main idea: <u>leanr how to aggregate feature information from a node's local neighborhood</u>
            - sample and aggregate approach: 
                - **[i]** Sample neighborhood (*each node aggregates the representations of the nodes in its immediate neighborhood into a single vector*)
                - **[ii]** Aggregate feature information from neighbors (*concatenate the node's current representation with the aggregated neighborhood vector -> fed through a fully connected layer with nonlinear activation function*)
                - **[iii]** predict graph context and label using aggregated information 
            
        - **[2]** train a set of **aggregator functions** -> learn to aggregate feature information from a node's local neighborhood
            - design an **unsupervised loss function** -> allow GraphSAGE to be trained without task-specific supervision
            - Aggregator architectures: `Mean aggregator`, `LSTM aggregator`, `Pooling aggregator`**LSTM** and **Pooling** -> **LSTM** and **Pooling** perform the best
        - **[3]** evaluate the algorithm on three node-classification benchmarks
    - <u>**Other Notes**</u>:
        - previous focused on embedding nodes from a **single** fixed graph, require all nodes in the graph are present during training => **transductive**, <u>do not naturally generalize to unseen nodes</u>
        - the inductive framework must learn to recognize **structural properties** of a node's **neighbourhood** -> reveal both <u>node's local role in the graph</u> and <u>its global position</u>
        - <u>Factorization-based embedding approaches</u> require expensive additional training after transductive learning; and the objective function is <u>invariant to orthogonal transformations of the embeddings</u>
    - <u>**Use cases**</u>:
        1. classifying academic papers into different subjects using the Web of Science citation dataset
        2. classifying Reddit posts as belonging to different communities
        3. classifying protein functions across various biological protein-protein interaction (PPI) graph
    - <u>**Further directions**</u>:
        1. extending GraphSAGE to incorporate **directed** or **multi-model** graphs
        2. exploring non-uniform neighborhood sampling functions
        3. learning these functions as part of the GraphSAGE optimization

---

(7) `Jul 2015` [Network Lasso: Clustering and Optimization in Large Graphs](https://arxiv.org/abs/1507.00280) *[David Hallac, Jure Leskovec, Stephen Boyd]*

* **Abstract**: 

> Convex optimization is an essential tool for modern data analysis, as it provides a framework to formulate and solve many problems in machine learning and data mining. However, general convex optimization solvers do not scale well, and scalable solvers are often specialized to only work on a narrow class of problems. Therefore, there is a need for simple, scalable algorithms that can solve many common optimization problems. In this paper, we introduce the **network lasso**, a generalization of the group lasso to a network setting that allows for simultaneous clustering and optimization on graphs. We develop an algorithm based on the **Alternating Direction Method of Multipliers (ADMM)** to solve this problem in a distributed and scalable manner, which allows for guaranteed global convergence even on large graphs. We also examine a non-convex extension of this approach. We then demonstrate that many types of problems can be expressed in our framework. We focus on three in particular - binary classification, predicting housing prices, and event detection in time series data - comparing the network lasso to baseline approaches and showing that it is both a fast and accurate method of solving large optimization problems.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[i]** formally define **network lasso**, a generalization of the group lasso to a network setting for simultaneous **clustering** & **optimization** on graph
            - network lasso problem: <u>**cost of node** + **edge cost** (sum of norms of differences of the adjacent edge varialbes)</u> *e.g. each vertex might represent the action of a control system*
            - the **edge** term are <u>regularization that encourages adjacent nodes to have close model parameters</u> -> `adjacent nodes should have similar models`
        - **[ii]** The author's **distributed** & **scalable** solution -> each vertex is controlled by one `agent`, they exchange information on the graph to solve the problem iteratively:
            - propose a easy to implement algorithm based on **Alternating Direction Method of Multipliers (ADMM)**
        - **[iii]** real examples
    - <u>**Other Notes**</u>:
        - With **large dataset** classical methods of convex methods fail due to **lack of scalability** -> <u>large-scale optimization</u> (*challenge: generalize, capable of scaling*)
        - **convex clustering**: a well studied instance of network lasso
    - <u>**Use cases**</u>:
        - problem of predicting house price
        > The network lasso solution empirically determines the neighbourhoods, so that each house can share a common model with houses in its cluster
    - <u>**Further directions**</u>:
        - the analysis of different non-convex functions phi
        - many ways to inmprove speed, performan and robustness: 
            1. *find closed-form solution for common objective function f(x)*
            2. *automatically determining the optimal ADMM parameter rho*
            3. *allow edge objective function beyond just the weighted network lasso*

---

(8) `May 2019` [Semi-Supervised Classification on Non-Sparse Graphs Using Low-Rank Graph Convolutional Networks](https://arxiv.org/abs/1905.10224) *[Dominik Alfke, Martin Stoll]*

* **Abstract**: 

> Graph Convolutional Networks (GCNs) have proven to be successful tools for semi-supervised learning on graph-based datasets. For **sparse** graphs, *linear and polynomial* filter functions have yielded impressive results. For **large non-sparse graphs**, however, network training and evaluation becomes prohibitively expensive. By introducing **low-rank filters**, we gain significant runtime acceleration and simultaneously improved accuracy. We further propose an architecture change mimicking techniques from Model Order Reduction in what we call a **reduced-order GCN**. Moreover, we present how our method can also be applied to **hypergraph datasets** and how **hypergraph convolution** can be implemented efficiently.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** introduction of **low-rank filters** -> decrease runtimes and produce more accurate classification
            - inspired by <u>Model Order Reduction</u> to accelerate the convolution operation
            - advantages of low-rank kernel matrices $K$: 
                - *[i]* the matrix products are much cheaper to evaluate
                - *[ii]* setting up it requires only a small number of eigenpairs
                - *[iii]* the dominant eigenvalues include clustering information -> damping noise to zero
        - **[2]** introduction of **pseudoinverse filter** -> better than standard linear filter
        - **[3]** **reduced-order GCN** -> dependent on the dataset, have good performance in hypergraph
            - define the **graph Laplacian** in hypergraph
    - <u>**Other Notes**</u>:
        - Effect method for semi-supervised learning (**SSL**): 
            - > **a small set of training data** and **clustering information** extracted from a vast amount of unlabeled data
        - **filter function space** is a crucial design choice in each GCN architecture
        - **[clustering]**: $i$-th  and $j$-th components of the vector are similar iff nodes $i$ and $j$ have a strong connection in the dataset graph
    - <u>**Use cases**</u>:
        - on **Hypergraph**, i.e. *[containing categorical data]*
        - good for data point classification
    - <u>**Further directions**</u>:

---

(9) `Feb 2019` [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) *[Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka]*

* **Abstract**: 

> Graph Neural Networks (GNNs) are an effective framework for representation learning of graphs. GNNs follow a neighborhood aggregation scheme, where the representation vector of a node is computed by *recursively aggregating and transforming representation vectors of its neighboring nodes*. Many GNN variants have been proposed and have achieved state-of-the-art results on both node and graph classification tasks. However, despite GNNs revolutionizing graph representation learning, there is **[limited understanding of their representational properties and limitations]**. Here, we present a theoretical framework for analyzing the expressive power of GNNs to **capture different graph structures**. Our results characterize the discriminative power of popular GNN variants, such as Graph Convolutional Networks and GraphSAGE, and show that they cannot learn to distinguish certain simple graph structures. We then develop a simple architecture that is provably the most expressive among the class of GNNs and is as powerful as the Weisfeiler-Lehman graph isomorphism test. We empirically validate our theoretical findings on a number of graph classification benchmarks, and demonstrate that our model achieves state-of-the-art performance.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** show that GNNs are <u>at most as powerful as the WL test</u> in *distinguishing graph structures*
            - intuitively a maximally powerfull GNN: <u>maps two nodes to the sam e location in the embedding space</u> --**only if**--> they have **identical subtree structures with identical features on the corresponding nodes**
        - **[2]** establish conditions on the **neighbor aggregation** and **graph readout function** => the resulting GNN is *as powerful as* the WL test
            - > ideal maximally powerful GNN: distinguish different graph structures by mapping them to different representations in the embedding space
        - **[3]** identify graph structure that cannot be identified by [popular GNN variants], `GCN` and `GraphSAGE`; chatacterize the kinds of structure that can be captured
            - **Less powerful** but still **Interesting** GNNs:
                - **1-layer perceptrons are not sufficient**
                - <u>Structures that confuse **mean** and **max** pooling</u> -> [have trouble distinguishing graphs with nodes that have **repeating features**]
                - **Mean pooling** method learns **DISTRIBUTION** of elements in a multiset (but not the exact multiset)
                - **Max pooling** learns **SETS with DISTINCT ELEMENTS** 
                    - *it is important to identify representative elements (or the skeleton) rather than to distinguish the exact structure or distribution*
        - **[4]** develop a <u>simple neural architecture</u> -> **Graph Isomorphism Network (GIN)** (*equal to the power of WL test*)

    - <u>**Other Notes**</u>:
        - design of new GNNs is mostly based on **empirical intuition**, **heuristics** and `experimental trial-and-error` => <u>there is little theoretical understanding of the prpperties and limitations of GNNs</u>
        - need to formally characterize how expressive different GNN => <u>represent and distinguish between different **graph structures**</u>
        - the more **discriminative** the multiset function is, the more powerful the representational power of the underlying GNN
        - **Graph Neural Networks**:-> follow a <u>neighborhood aggregation</u>
            - `GraphSAGE` use **element-wise max-pooling**; `GCN` use **element-wise mean pooling**
        - **Weisferler-Lehman test**:
            - > **WL test of graph isomorphism**: an effective and computationally efficient test that distinguishes a broad class of graphs
            - the test iteratively: 
                1. aggregates the lables of **nodes** and their **neighborhoods**
                2. hashes the aggregated labels into unique new lables
            - it decides two graphs are **non-isomorphic** -> if <u>at some iteration the labels of the nodes between the two graphs differ</u>
        - **Multiset**: allows multiple instaances for its elements
        - GNN with **sum aggregators** tend to fit the **training sets** better than those with mean and max pooling aggregators
        - WL kernel is **not able to learn how to combine node featues**
    - <u>**Use cases**</u>:
        - `molecules networks`, `social networks`, `biological networks`, `financial networks`
        - `node classification`, `graph classification`, `link prediction`
    - <u>**Further directions**</u>:
        1. apply the framework to analyze and understanding <u>other aggregation schemes</u>
        2. go beyond neighborhood aggregation (or message passing) -> <u>pursue possibly even more powerful architectures for learning with graphs</u>
        3. understand and improve the **generalization properties** of GNNs as well as better understanding their **optimization** landscape

---

(10) `May 2017` [Learning Edge Representations via Low-Rank Asymmetric Projections](https://arxiv.org/abs/1705.05615) *[Sami Abu-El-Haija, Bryan Perozzi, Rami Al-Rfou]* `Graph likelihood`

* **Abstract**: 

> We propose a new method for **embedding graphs** while preserving **directed** edge information. Learning such continuous-space vector representations (or embeddings) of nodes in a graph is an important first step for using network information (from social networks, user-item graphs, knowledge bases, etc.) in many machine learning tasks.
Unlike previous work, we 
    
* > (1) explicitly model an edge as a function of node embeddings
* > (2) propose a novel objective, the "**graph likelihood**", which contrasts information from sampled random walks with non-existent edges. 

> Individually, both of these contributions improve the learned representations, especially when there are memory constraints on the total size of the embeddings. When combined, our contributions enable us to significantly improve the state-of-the-art by learning more concise representations that better preserve the graph structure.
We evaluate our method on a variety of **link-prediction** task including social networks, collaboration networks, and protein interactions, showing that our proposed method learn representations with error reductions of up to 76% and 55%, on directed and undirected graphs. In addition, we show that the representations learned by our method are quite space efficient, producing embeddings which have higher structure-preserving accuracy but are 10 times smaller.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** explicitly model a **directed edge function** -> **[low-rank affine projects]** on a manifold that is produced by a DNN (`use DNN to map nodes onto a low-dimensional manifold`) + (`define a function between two nodes as a projection in the manifold coordinates`)
            - **[i]** $f()$ removes <u>degrees of freedom</u> as it passes an embedding through the DNN activation functions 
            - **[ii]** reduces overfitting and improve generalization (*as f can contrain the embeddings with less degrees of freedom*)
            - **[iii]** hidden layers inf find correlations in the data, as many graphnodes have similar connections
        - **[2]** propose a new objective function, **the graph likelihood** -> <u>joinly maximizing the edge function and the manifold</u> (*inspired from MLE in logistic regression*)
        - **[3]** improve the SOTA on learning continuous graph representation, especailly on **directed graphs** while producing significantly **smaller** representation
        - learn an asymmetric transformation of the nodes -> combined for any pair of nodes to moedel the strength of their directed relationships
        - Adjacency (non-embedding) Baselines: `Jaccard Coefficient`, `Common Neighbors`, `Adamic Adar`
        - Embedding methods: `Laplacian EigenMaps`, `node2vec`, `DNGR`
    - <u>**Other Notes**</u>:
        - <u>continuous space representations</u>: leanr a vector space that highly preserve the graph structure
        - traditional ***eigen*** methods learn embeddings that `minimize the euclidean distance of the connected nodes`
        - shortcomings of embedding method (like random wlak):
            - do not explictly model deges
            - unable to capture asymmetic relationships (different direction of edges)
    - <u>**Use cases**</u>:
        * **link-prediction** in `social network`, `collaboration network` and `protein interaction`
        * *followers* and *followees* 
    - <u>**Further directions**</u>:
        - further investigation for learning continuous representation of graphs

---

(11) `Mar 2014` [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652) *[Bryan Perozzi, Rami Al-Rfou, Steven Skiena]* `DeepWalk` 

* **Abstract**: 

> We present DeepWalk, a novel approach for learning latent representations of **vertices** in a network. These latent representations *encode social relations in a continuous vector space*, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs. DeepWalk uses **local information** obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences. We demonstrate DeepWalk's latent representations on several multi-label network classification tasks for social networks such as `BlogCatalog`, `Flickr`, and `YouTube`. Our results show that DeepWalk outperforms challenging baselines which are allowed a global view of the network, especially in the presence of missing information. DeepWalk's representations can provide F1 scores up to 10% higher than competing methods when labeled data is sparse. In some experiments, DeepWalk's representations are able to outperform all baseline methods while using 60% less training data. DeepWalk is also scalable. It is an **online learning** algorithm which builds useful incremental results, and is trivially parallelizable. These qualities make it suitable for a broad class of real world applications such as network classification, and anomaly detection.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** introduce <u>deep learning</u> to analyze graphs, introduce **DEEPWALK**
            - use DL to <u>robust representations</u> -> statistical learning
            - DEEPWALK (**Random Walk Generator** + **Update Procedure**)learns `structual regularities` with *short random walk* -> capture network topology information
            - Learning **label independent representations** of the graph -> shared among tasks
        - **[2]** evaluate the representations on **multi-label classfication tasks** on several social networks
        - **[3]** demonstrate the **scalability** of the algorithm
    - <u>**Other Notes**</u>:
        - The sparsity of a network representation is both a strength and a weakness
            - > **+**: enables the design of efficient discrete algorithm
            - > **-**: harder to generalize in statistical learning
        - <u>**social representation**</u>: *latent features of the vertices that capture neighborhood similarity and community membership* 
        - Traditional approches to **realational classification**: 
            - inference in an **undirected Markov network** -> use **iterative approximate inference algorithms** (e.g. `Gibbs Sampling`, `label relexation` -> posterior distribution of labels)
        - <u>Characteristics</u> for learning social representation: 
            - > `Adaptability`, `Community aware`, `Low dimensional`, `Continuous`(-> have smooth decision boundaries between communities which allows more robust classification)
        - `model natural language` -> `model community structure in networks`
        - the relaxations of the optimization for target: 
            1. the order independence assumption -> better captures a sense of **nearness** that is provided by <u>random walk</u>
            2. speeding up the training time by building small models as one vertex is given at a time
    - <u>**Use cases**</u>:
        - <u>classifiy members of a social network into one or more categories</u>
    - <u>**Further directions**</u>:
        1. focus on investigating duality further
        2. improve language modeling
        3. strengthening the theoretical justifications of the method

---

(12) `Aug 2018` [A Tutorial on Network Embeddings](https://arxiv.org/abs/1808.02590) *[Haochen Chen, Bryan Perozzi, Rami Al-Rfou, Steven Skiena]*

* **Abstract**: 

> Network embedding methods aim at learning **low-dimensional latent representation** of nodes in a network. These representations can be used as features for a wide range of tasks on graphs such as classification, clustering, link prediction, and visualization. In this survey, we give an overview of network embeddings by summarizing and categorizing recent advancements in this research field. 
> We first discuss the **[1] desirable properties** of network embeddings and briefly introduce the **[2] history** of network embedding algorithms. Then, we discuss network embedding methods under **[3]different scenarios**, such as *supervised versus unsupervised learning*, *learning embeddings for **homogeneous** networks versus for **heterogeneous** networks*, etc. We further demonstrate the **[4]applications** of network embeddings, and conclude the survey with **[5]future work** in this area.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **Brief history of Network Embedding**: (lower performance compared to **Deep Learning methods**)
            - **[1] PCA and Multidimensional Scaling (MDS)**: <$O(n^3)$>
                - > **MDS**: project each row of M to a k-dim components -> the distance between different objects in the oringinal feature matrix M is best preserved in the k-dim space
                - The matrix to factorize could be `adjacency matrix`, `normalized Laplacian matrix`, `all-pairs shortest path matrix`
                - Both PCA & MDS **fail to discover the non-linearity** and **have high time complexity**
            - **[2] IsoMap and Locally Linear Embeddings (LLE)**: -> **non-linear**
                - > **IsoMap**: extension of MDS, *preserving geodesic distances in the neighborhood graph of input data* (neighborhood of **node i** contructed by connecting [nodes closer than a threshold]/[nodes which are k-nearest neighbors])
                - > **LLE**: only exploits the local neighborhood of data point, not estimate distance between distant data points
                - **time complexity** to large
            - **[3] Laplacian eigenmaps (LE)**: -> use **spectral properties (eigenvectors)**
                - > **LE**: represent each node by [eigenvectors associated with its **k-smallest** nontrivial eigenvalues]
            - **[4] SocDim**: using the spectral properties of the **[modularity matrix]** as latent social dim in the network
        - **With Deep Learning method**: **Deep walk**
            - > **Advantanges**: [1] can be generated on demand; [2]scalable **[3] intraoduce a paradigm for deep learning on graphs**
            - <u>Deep Walk Paradigm</u>: can be expanded in `complexity of graphs`, `complexity of methods`
                - **[i]** Choose a matrix associated with the input graph
                - **[ii]** **Graph Sampling**: Sample sequences from the chose matrix 
                    1. save time by approximating the matrix
                    2. sequence of symbols are much more easier for deep learning models to deal with
                - **[iii]** Learn embeddings from the sequences or the matrix itself
                - **[iv]** ouptut node embeddings
        - **Unsupervised** Network Embedding:
            - <u>Summary of unsupervised network embedding methods</u> **[in simple undirected graphs]**
                - `Deep Walk`, `LINE`, `Node2vec`, `Walklets`, `GraRep` -> all have hyper-parameter
                - `GraphAttention` -> learn the attention over the power series of the graph transition matrix 
                    - > learns a multi-scale representation which best predicts links in the original graph
                - `SDNE`, `DNGR` -> combined with autoencoder
            - <u>Directed Graph Embedding</u>: `HOPE` [Learning Edge Representations via Low-Rank Asymmetric Projections](https://arxiv.org/abs/1705.05615)
            - <u>Edge Embeddings</u>: -> `link prediction` [Learning Edge Representations via Low-Rank Asymmetric Projections](https://arxiv.org/abs/1705.05615)
                - > learn edge representations via low-rank asymmetric projections
            - <u>Signed Graph Embeddings</u>: `SiNE` & `SNE`
                - > **SiNE**: maximizing the margin between the embedding similarity of friends (+1 connected) and the embedding similarity of foes (-1 connected)
                - > **SNE**: predicts the representation of a target node by linearly combines the representation of its context nodes; [tow signed-type vector are incorporated into the log-bilinear model]
            - <u>Subgraph Embeddings</u>: 
                - `Deep graph kernel`: *general framework for modelling sub-structure similarity in graphs*
            - <u>Meta-strategies for Improving NEtwork Embeddings</u>
                - [weakness of neural methods for network embeddings]:
                    1. all **local approaches** -> limited to strucuture immediately around a node, **fail to uncover important long distance global structural pattern**
                    2. all rely on **non-convex optimization goal**
                - [HARP](https://arxiv.org/abs/1706.07845): `embedding capture both the local and global structures of the ven graphs`
        - **Attributed Network Embedding**: Desirable to learn from `node attributes` and `edge attributes`
            - **[1]** textual attrigutes
                - > **TADW**: incorporates the text features into the matrix factorization process
                - Jointly model network structure and node features -> `enforce the embedding similarity between nodes with similar feature vectors`
                    - > **CENE**: treats text content as a special type of node -> [node-node links] & [node-content links] for node embedding
                - > **HSCA**
            - **[2]** node labels: e.g. `citation network`: *venue* or *year of publication*
                - > **GENE**: also predicts the group information of context nodes as a part of the optimization goal
                - [Community Preserving Network Embedding](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14589): preserves the **community structures within network**
            - **[3]** semi-supervised network embedding methods: 
                - [Planetoid](https://arxiv.org/abs/1603.08861)
                - [Max-margin DeepWalk (MMDW)](https://www.ijcai.org/Proceedings/16/Papers/547.pdf)
        - **Heterogeneous Network Embedding** *(jointly minimizing the loss over each modality)*
            - [Heterogeneous Network Embedding via Deep Architectures](http://www.ifp.illinois.edu/~chang87/papers/kdd_2015.pdf): `feature representation for each modality` -> `map them to same embedding spae`
            - [Representation Learning for Measuring Entity Relatedness with Rich Information](https://pdfs.semanticscholar.org/ff66/50ee2efab6f6ec155ecb644a329397cb16fa.pdf)
            - [Learning multi-faceted representations of individuals from heterogeneous evidence using neural networks](https://arxiv.org/abs/1510.05198) -> neural network model for learning user representation in a heterogeneous social network
            - **HEBE**:  embedding large-scale heterogeneous event network
            - **EOE**: for coupled heterogeneous network (two networks connected by inter-network edges)
            - **Metapath2vec**: -> [extending random walks and embeding learning methods to hetero-networks]

    - <u>**Other Notes**</u>:
        - due to **[billions of nodes and edges in the information network can be intractable to perform complex inference procedures]** -> use **network embedding** to solve this problem
            * > find a mapping function: [converts each node in the nework to a low-dimenstional latent representation] (can be use as features)
        - <u>Target network embedding characteristics</u>: 
            - **[1] Adaptability**: *new application should not require repeating learning process*
            - **[2] Scalability**: *able to process large-scale networks in a <u>short period of time</u>*
            - **[3] Community aware**: *distance between latent representations should represent a **metric** for evaluating **similarity** between the corresponding members of the network* -> `generalization in networks with homophily`
            - **[4] Low dimensional**: -> better and speed up convergence and inference
            - **[5] Continuous**: to model partial community membership in continuous space, <u>continuous representation has smooth decision boundatires between communities</u> -> **Robust Classification**
        - **Heterogeneous Network**: `network with multiple type of nodes or multiple types of edges`
        - **Signed Graph**: `edge is assigned with a weight from {+1,-1}` -> could be used to reflect **agreement or trust**

    - <u>**Use cases and application**</u>:
        - <u>Knowledge Representation</u>: *[encoding facts about the world using short sentences composed of (subjects, predicates and objects)]*
            - `GenVector`: learning social knowledge graphs
            - `RDF2Vec`(Resource Description Framework)
        - <u>Recommender Systems</u>:
            - > interactins between (**users**, **users' queries**, **items**) -> form a heterogeneous network to encode the latent preference of users over time
            - [Query-based Music Recommendations via Preference Embedding](https://dl.acm.org/citation.cfm?id=2959169) -> embed user preference and query intention into low-dimensional vector space
        - <u>NLP</u>:
            - `PLE` (label noise reduction in entity typing), `CANE`(context-aware netowrk embedding frame work), `community-based QA framework`
    - <u>Social Network Analysis</u> `refer back to the paper`
        - *predicting the exact age of users in social network*
        - modelling social network and mobile trajectories simultaneously
        - Obtain conect embedding of higher quality, by learning **wikipedia page representations**
        - align usersacross different social network
        - measuring similarity between historical figures
        - browsing through large lists in the absense of a predefined hierachy

    - <u>**Further directions**</u>:
        - **[Problem 1]**: most of the strategies relies on a **rigid definition of context nodes indentical for all networks** -> `unifying different network embedding under a general framework`, only **Graph Attention** have compacity for different networks
        - **[Problem 2]**: dependence upon general loss function and optimization models, suboptimal compared to `end2end embeddings methods designed specifically for a task`
            - > design loss functions and optimization models for a specific task


---

(13) `Feb 2018` [Link Prediction Based on Graph Neural Networks](https://arxiv.org/abs/1802.09691) *[Muhan Zhang, Yixin Chen]*

* **Abstract**: 

> **Link prediction** is a key problem for network-structured data. Link prediction heuristics use some score functions, such as **common neighbors** and **Katz index**, to measure the **likelihood** of links. They have obtained wide practical uses due to their **simplicity, interpretability, and for some of them, scalability**. However, every heuristic has a **strong assumption** on *when two nodes are likely to link*, which limits their effectiveness on networks where these assumptions fail. In this regard, a more reasonable way should be learning a suitable heuristic from a given network instead of using predefined ones. By extracting a **local subgraph** around each target link, we aim to learn a function mapping the subgraph patterns to link existence, thus automatically learning a heuristic' that suits the current network. In this paper, we study this heuristic learning paradigm for link prediction. 

> **[1]**First, we develop a novel **γ-decaying heuristic theory**. The theory unifies a wide range of heuristics in a single framework, and proves that all these heuristics can be well approximated from local subgraphs. Our results show that local subgraphs reserve rich information related to link existence. 

> **[2]**Second, based on the γ-decaying theory, we propose a new algorithm to learn heuristics from local subgraphs using a graph neural network (GNN). Its experimental results show unprecedented performance, working consistently well on a wide range of problems.

* **Key notes**: 
    - <u>**Other Notes**</u>:
        - **[link prediction]**: predict whether two nodes in a network are likely to have a link
        - **[Heuristic methods]**: compute some heuristic node similarity scores as the likelihood of links -> *can be categorized based on the <u>maximum hop of neighbors</u>*
            - **First-order heuristics**: `common neighbors (CN)`, `preferential attachment (PA)`
            - **Second-order heuristics**: `Adamic-Adar (AA)`, `Resource allocation (RA)`
            - **[h-order heuristics]**: which require knowing up to h-hop neighborhood of the target nodes
            - > some **high order heuristics** require knowing the entire network e.g. `Katz`, `PageRank (PR)`, `SimRank (SR)`
        - Heuristic methods have **strong assumptions** on when links may exist e.g. `two nodes are more likely to connect if they have many common neighbors` -> <u>might cause problem if the assumption fail in specific network</u>
        - **Graph structure features (include heuristics)**: features located inside the observed node and edge structure of the network -> can be calculated directly from the graph
        - [Weisfeiler-Lehman Neural Machine (WLNM)](https://www.cse.wustl.edu/~muhan/papers/KDD_2017.pdf): use <u>fully connected neural network</u> to learn <u>which enclosing subgraphs correspond to link existence</u> - learn heuristic automatically 
            - > **problem 1**: high order heuristic have much better performance but need entire network as subgraph -> which means unaffordable time and memory consumption
            - > **problem 2**: fully connected layer -> fixed size tensors -> information loss due to truncate
            - > **problem 3**: due to limitation of **adjacency mnatrix representatinon** -> cannot combine latent & explicit features
            - > **problem 4**: lack of theoretical justification
        - **Latent feature**: use matrix representation of the network to learn low-dimentional representation/embedding for each node; **explicit features**: node attributes, describing all kinds of side information about individual nodes
            - > combine both with graph structure feature could improve performance

    - <u>**Main contributions**</u>: 
        - **[1]** new theory (proof) for **learning link prediction heuristics** - **learn from local enclosing subgraph** 
            - using **$\gamma$-decaying theory** to effectively approximated from an h-hop enclosing subgraph
            - from it we are able to accurately calculate 1st and 2nd order heuristic, and approximate a wide range of high-order heuristics with small errors
        - **[2]** Novel link prediction framework **SEAL**: to learn general graph structure features (heuristics) from **local** enclosing subgraph as input
            - Use **GNN** (graph convolusion layer) instead of *fully-connected NN*
            - SEAL permits not only **subgraph structures** but aslo **latent and explicit** node features -> refer to <u>**<Other Notes>**</u>
                - concatenate the features to $X$
                - use negative injection
            - Include 3 steps: 
                - **[1]** enclosing subgraph extraction for a set of sampled positive (observed) links and set of sampled negative (unobserved) links
                - **[2]** node information matrix construction;
                - **[3]** GNN learning -> adjacency matrix + node information matrix
            - <u>node labelling</u> is important -> **let GNN tell where are the target nodes between which a link existence should be predicted**
                - the authors propose a **Double-Radius Node Labeling (DRNL)**, it has a perfect **hashing function** -> allows fast closed-form computations
                > iteratively assign larger labels to nodes with a **larger radius** wrt both center nodes

    - <u>**Use cases**</u>:
        1. friend recommendation
        2. movie recommendation
        3. knowledge graph completion
        4. metabolic network reconstruction

    - <u>**Further directions**</u>:
        - knowledge graph completion
        - recommender system

---

(14) `Sep 2018` [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) *[Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, R Devon Hjelm]*

* **Abstract**: 

> We present **Deep Graph Infomax (DGI)**, a general approach for learning node representations within graph-structured data in an **unsupervised** manner. DGI relies on **maximizing mutual information** between patch representations and corresponding high-level summaries of graphs---both derived using established graph convolutional network architectures. The learnt patch representations summarize subgraphs centered around nodes of interest, and can thus be reused for downstream node-wise learning tasks. In contrast to most prior approaches to unsupervised learning with GCNs, DGI does **not rely on random walk objectives**, and is readily applicable to *both transductive and inductive* learning setups. We demonstrate competitive performance on a variety of node classification benchmarks, which at times even exceeds the performance of supervised learning.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** **DGI**(Deep Graph InfoMax) propose an anternative objective for unsupervised graph learning -> based on **mutual information**
            - Mutual Information Neural Estimation (MINE) -> make scalable estimation of mutual information possible and practical 
                - > training a statistics network as a **classifier** of samples coming from the joint distribution of two random variables and their product of marginals
            - **DIM**(Deep InfoMax) -> learn representation of high-dim data
            - <u>combine the above two, adapt ideas from DIM to graph domain</u> (use a *noice-constrastive* type objective with a standard binary cross-entropy (BCE) loss between samples from the joint and the product of marginal)
            - DGI is **Contrastive**: objective is based on classifying local-global pairs and negative-sampled counterparts
            - the authors' method contrast global & local parts simultaneously -> <u>global variable is computed from all local variables</u>
        - **[2]** Steps of **DGI**: 
            1. sample a negative example by using the corruption function
            2. obtain patch representations for the **input graph** by passing it through the encoder
            3. obtain path representations for the **negative example** by passing it through the encoder
            4. summarize the input graph by passing its patch representations through the **readout function**
            5. Update parameters by applying gradient descent ot maximieze the objective function


    - <u>**Other Notes**</u>:
        - unsupervised representation learning with graph-structured data rely on **random walk-based objectives**
        - Limitations of <u>random walk-based methods</u>:
            1. over-emphasize proximity information at the expense of structural information
            2. performance is highly dependent on hyperparameter choices
            3. enforce an inductive bias that neighboring nodes have similar representations
        - **Contrastive method**: train an encoder to be **contrastive** between representations that capture statistical dependencies of interest and those that do not
        - **Sampling strategies**: 
            - previous get positive samples from *short random walks in the graph*
            - recent work adapt **node-anchored sampling** to use a **curriculum-based negative sampling** scheme
        - **Predictive coding**:
            - previous Contrastive predictive coding (**CPC**) all predicitve -> the contrastive objective effectively trains a predictor between structurally-specified parts of the input
            - others (encoding objectives on adjacency matrix; incorporation of community-level constraints into node embeddings) rely on **matrix factorization-style losses** ---> **not scalable to large graph**
    - <u>**Use cases**</u>:
        - **[i]** classifying research papers into topics on the `Cora`, `Citeseer` and `Pubmed` **citation networks**
        - **[ii]** predicting the community structure of a social network modeled with `Reddit` posts
        - **[iii]** classifying protein roles within **protein-protein interaction** (PPI) networks (*requiring generalisation to the unseen networks*) 
    - <u>**Further directions**</u>:

---

(15) `Nov 2016` [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) *[Thomas N. Kipf, Max Welling]*

[*Tutorial*](https://towardsdatascience.com/tutorial-on-variational-graph-auto-encoders-da9333281129)

* **Abstract**: 

> We introduce the variational graph auto-encoder (VGAE), a framework for unsupervised learning on graph-structured data based on the variational auto-encoder (VAE). This model makes use of latent variables and is capable of learning interpretable latent representations for undirected graphs. We demonstrate this model using a graph convolutional network (GCN) encoder and a simple inner product decoder. Our model achieves competitive results on a link prediction task in citation networks. In contrast to most existing models for unsupervised learning on graph-structured data and link prediction, our model can naturally incorporate node features, which significantly improves predictive performance on a number of benchmark datasets.

* **Key notes**: 
    - <u>**Contributions**</u>: 
        - Introduced **variational graph VAE**: -> 
        > learn interpretable latent representation for **undirected** graphs
            - **GCN encoder** -> **inner product decoder**
        - use model for link prediction task in citation network
    - <u>**Other notes**</u>:
        - Adding input features significantly improves predictive perfomance across datasets
    - <u>**Further direction**</u>:
        - investigate better-suited prior ditribution other than **Gaussian prior**
        - more flexible generative models
        - the application of a stochastic gradient descent algorithm for improved scalability

---

(16) `Apr 2019` [DeepGCNs: Can GCNs Go as Deep as CNNs?](https://arxiv.org/abs/1904.03751) *[Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem]*

* **Abstract**: 

> Convolutional Neural Networks (CNNs) achieve impressive performance in a wide variety of fields. Their success benefited from a massive boost when very deep CNN models were able to be reliably trained. Despite their merits, CNNs **[fail to properly address problems with non-Euclidean data**. To overcome this challenge, Graph Convolutional Networks (GCNs) build graphs to represent non-Euclidean data, borrow concepts from CNNs, and apply them in training. GCNs show promising results, but they are usually **[limited to very shallow models due to the vanishing gradient problem]**. As a result, most state-of-the-art GCN models are no deeper than 3 or 4 layers. In this work, we present new ways to successfully train very deep GCNs. We do this by borrowing concepts from CNNs, specifically **residual/dense connections** and **dilated convolutions**, and adapting them to GCN architectures. Extensive experiments show the positive effect of these deep GCN frameworks. Finally, we use these new concepts to build a very deep 56-layer GCN, and show how it significantly boosts performance (+3.7% mIoU over state-of-the-art) in the task of point cloud semantic segmentation. We believe that the community can greatly benefit from this work, as it opens up many opportunities for advancing GCN-based research.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** adapt **residual/dense** connections and **dilated convolustions** to GCNs
            - borowing orthogonal tricks from CNN
            - `ResNet`: <u>add residual connections between inputs and ouputs of layers</u>
                - <u>propose a **graph residual learning framework**</u> -> *learns an underlying mapping H by fitting another mapping F*
            - `DenseNet`: <u>apply resnet further and add connections across layers</u>, 
                - exploit dense connectivity among layers -> **[improves information flow in the network and enables efficient reuse of features among layers]**
            - `Dilated Convolutions`: <u>increasing the receptive field without loss of resolution</u>
                - propose to use **Dilated k-NN** -> <u>to find dilated neighbors after every GCN layer and constrct a Dilated Graph</u>
        - **[2]** present extensive experiments on **point cloud data** -> showing the effect of each of these added new layers to the **stability** and **performance** of training deep GCNs
            - use **point cloud semantic segmentation** as experimental testbed
            - Network structure: **GCN backbone block** + **fusion block** (*fuse the global and multi-scale local features*) + **MLP prediction blck**
        - **[3]** show how these new concepts help build a **56-layer GCN** -> achieve close to 4% boost in performance
    - <u>**Other Notes**</u>:
        - Reasons for increased interests in **GCN**:
            1. the increasing proliferation of **non-Euclidean** data in real-world applications
            2. the limited performance of CNN when dealing with non-Euclidean data
        - Classic problem: <u>stacking more layers in to a GCN leads to the common **vanishing gradient** problem</u> (*normally no deeper than 4 layers*)
        - **Graph Convolution Network**
            - represent **vertices** by -> **associating each vertex with a feature vector**
            - graph convolution operation: **aggregation function** + **update operations**
                - <u>aggregation function</u>: **compile information from the neighborhood of vertices**
                    - > e.g. `mean aggregator`, `max-pooling aggregator`, `attention aggregator`, `LSTM aggregator`
                - <u>update function</u>: **performa a non-linear transform on the aggregated information** -> **to compute new vertex representations**
                    - > e.g. `MLP`, `gated network` etc.
        - **Dynamic Edges**: <u>graph structure is allowed to change in each layer</u>
            - **ECC (Edge-Conditioned Convolution)**: *uses dynamic edge-conditional filters to learn an <u>edge-specific weight matrix</u>*
            - **Graph-Convolution GAN**: use k-NN graphs to construct the neighbourhood of each vertex in every layer -> *generate point clouds*
            - > We find that **dynamically changing neighbourhood** in GCNs helps **alleviate** the **over-smoothing** problem and results in an effectively **large receptive field**
        - **Chanllenges in Point Cloud segmentation**: <u>unordered and irregular structure of 3D point clouds</u> -> [3D spatial coordinates and possibly auxiliary features such as color and surface normal]
    - <u>**Use cases**</u>:
        - <u>predict individual relations in social networks</u>
            - Graph represent connections between individuals based on **mutual interests/relations** - in social network
            - GCN help better estimate **edge strength**
        - <u>model proteins for drug discovery</u>
            - Graphs can model chemical molecule structures
        - <u>enhance predictions of recommendation engines</u>
            - accurate modelling of user interatctions -> improved product recommendations
        - <u>**point cloud semantic segmentation**</u>
        - [Graph] representation in NLP -> <u>complex relations between large text units</u>
        - [CV] <u>scene graph generation</u>: semantic relations between objects are modelled using a graph
            - also predict semantic relations between object pairs
        - [CV] image is reconstructed given a graph representation of the scene
        - [CV] model human joints for action recognition in video
        - [CV] 3D point cloud processing
        - directly processing unordered point cloud representation
        - **Column Netwok (CLN)** -> <u>collective classifcation in relational learning</u>
        - **Highway GCN** -> for user geo-location in social media graphs
        - **Jump Knowledge Network** -> representation learning

    - <u>**Further directions**</u>:
        1. explore how to transfer: 
            - other operators (e.g.`deformable confolutions`)
            - other architectures (e.g. `feature pyramid architectures`)
        2. study different distince meatures to compute dilated k-NN
        3. constructing graphs with different k at each layer
        4. better dilation rate schedules
        5. combining residual and dense connections
        6. more suitable sampling approach -> should lead to further performance gains on the task 

---

(17) `Jun 2019` [vGraph: A Generative Model for Joint Community Detection and Node Representation Learning](https://arxiv.org/abs/1906.07159) *[Fan-Yun Sun, Meng Qu, Jordan Hoffmann, Chin-Wei Huang, Jian Tang]*

* **Abstract**: 

> This paper focuses on two fundamental tasks of graph analysis: **community detection** and **node representation learning**, which capture the [global] and [local] structures of graphs, respectively. In the current literature, these two tasks are usually independently studied while they are actually highly correlated. We propose a probabilistic generative model called vGraph to learn community membership and node representation collaboratively. Specifically, we assume that each node can be represented as a mixture of communities, and each community is defined as a multinomial distribution over nodes. Both the mixing coefficients and the community distribution are parameterized by the low-dimensional representations of the nodes and communities. We designed an effective **variational inference algorithm** which regularizes the *community membership of neighboring nodes to be similar in the latent space*. Experimental results on multiple real-world graphs show that vGraph is very effective in both community detection and node representation learning, outperforming many competitive baselines in both tasks. We show that the framework of vGraph is quite flexible and can be easily extended to detect **hierarchical communities**.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** propose a novel **probabilistic generative** model called **vGraph** -> <u>joint community detection and node representation learning</u>
            - > **Assumption 1**: each node can be represented by a mixture of multiple communities and described by a **multinomial distribution** over communities `p(z|v)`
            - > **Assumption 2**: each community is modeled as a distribution over the nodes `p(v|z)`
            - By this approach: <u>it allows the node representation and the communities to interact in a mutually beneficial way</u>
            - **scalable**, **can be optimized efficiently**
            - [parameterizes the node-community distributions by introducing node and community embeddings]
        - **[2]** design a very effective algorithm for inference and back propagation
            - use **variaonal inference** -> maximizing the lower-bound of the data likelihood
            - use `Gumbel-Softmax` tricks -> to obtain gradients for the evidence lower bound
            - added a **smoothness regulatization term** to the objective function -> ensure that **[community membership of neighboring nodes is similar]**
        - **[3]** show that the model can be easily extended to detect **hierarchical communities**
            - at every node of the tree -> there is an embedding vector associated with the community
            - similar to the hierarchical softmax parameterization used in language models
    - <u>**Other Notes**</u>:
            - > Graphs or networks, are a general and flexible data structure to encode complex replationships among objects
        - **Community detection**: 
            - > *aims to cluster nodes into mulitple groups called communities*
            - each community contain nodes that are <u>more closely connected to each other than to nodes in different communities</u>
            - classical method - `spectral cluster`: 
                - > **Assumption**: nieghboring nodes tend to belong to the same communities
                -  detect communitites by find the **eigenvectors** of the **graph Laplacian**
            - use <u>matrix factorization techniques</u> -> *to recover the node-community affiliation matrix by performing a low-rank decomposition of the graph adjacency matrix*
                - but `not scalable`, `restricted by capacity of bi-linear model`
        - **node representation Learning**: * describe nodes using low-dimensional features*
            - > **Goal**: learn distributed representations of nodes in graphs so that nodes with similar local connectivity tend to have similar representations
            - classical method: `focus on local`, `ignore global community information`
                - **DeepWalk**:  a method that adopts truncated random walk and Skip-Gram to learn vertex embedding 
                - **Node2Vec**: adopts biased random walk and Skip-Gram to learn vertex embeddings
        - **Clustering**: *captures **Global** structure of graphs* -> `exploratory analysis`
        - intuition to combine <u>clustering</u> and <u>node embedding</u>:
            1. node representations can be used as good features for community detection (*through K-means*)
            2. node community membership can provide **good contexts** for learning node representations
    - <u>**Use cases**</u>:
        1. overlapping community detection
        2. non-overlapping comminity detection
        3. node classfication
    - <u>**Further directions**</u>:

---

(18) `Nov 2018` [Bayesian graph convolutional neural networks for semi-supervised classification](https://arxiv.org/abs/1811.11103) *[Yingxue Zhang, Soumyasundar Pal, Mark Coates, Deniz Üstebay]*

* **Abstract**: 

> Recently, techniques for applying convolutional neural networks to graph-structured data have emerged. Graph convolutional neural networks (GCNNs) have been used to address node and graph classification and matrix completion. Although the performance has been impressive, the current implementations have limited capability to incorporate uncertainty in the graph structure. Almost all GCNNs process a graph as though it is a ground-truth depiction of the relationship between nodes, but often the graphs employed in applications are themselves derived from noisy data or modelling assumptions. Spurious edges may be included; other edges may be missing between nodes that have very strong relationships. In this paper we adopt a Bayesian approach, viewing the observed graph as a realization from a parametric family of random graphs. We then target inference of the joint posterior of the random graph parameters and the node (or graph) labels. We present the Bayesian GCNN framework and develop an iterative learning procedure for the case of assortative mixed-membership stochastic block models. We present the results of experiments that demonstrate that the Bayesian formulation can provide better performance when there are very few labels available during the training process.

* **Key notes**: 
    - 

---

(19) `May 2017` [Supervised Community Detection with Line Graph Neural Networks](https://arxiv.org/abs/1705.08415) *[Zhengdao Chen, Xiang Li, Joan Bruna]*

* **Abstract**: 

> We study data-driven methods for community detection on graphs, an inverse problem that is typically solved in terms of the spectrum of certain operators or via posterior inference under certain probabilistic graphical models. Focusing on random graph families such as the stochastic block model, recent research has unified both approaches and identified both statistical and computational signal-to-noise detection thresholds. This graph inference task can be recast as a node-wise graph classification problem, and, as such, computational detection thresholds can be translated in terms of learning within appropriate models. We present a novel family of Graph Neural Networks (GNNs) and show that they can reach those detection thresholds in a purely data-driven manner without access to the underlying generative models, and even improve upon current computational thresholds in hard regimes. For that purpose, we propose to augment GNNs with the non-backtracking operator, defined on the line graph of edge adjacencies. We also perform the first analysis of optimization landscape on using GNNs to solve community detection problems, demonstrating that under certain simplifications and assumptions, the loss value at the local minima is close to the loss value at the global minimum/minima. Finally, the resulting model is also tested on real datasets, performing significantly better than previous models.

* **Key notes**: 
    - 

---

(20) `Oct 2018` [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997) *[Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann]*

* **Abstract**: 

> **Neural message passing** algorithms for **semi-supervised** classification on graphs have recently achieved great success. However, for classifying a node these methods only consider nodes that are a few propagation steps away and the size of this utilized neighborhood is hard to extend. In this paper, we use the relationship between graph convolutional networks (GCN) and PageRank to derive an **improved propagation scheme** based on **personalized PageRank**. We utilize this propagation procedure to construct a simple model, personalized propagation of neural predictions (PPNP), and its fast approximation, APPNP. Our model's training time is on par or faster and its number of parameters on par or lower than previous models. It leverages a large, adjustable neighborhood for classification and can be easily combined with any neural network. We show that this model outperforms several recently proposed methods for semi-supervised classification in the most thorough study done so far for GCN-like models. Our implementation is available online.

* **Key notes**: 
    - **<u>Main Contribution</u>**:
        - Highlight the inherent connectin between the **limited distribution** and **PageRank**
        - Propose an algorithm -> <u>utilizes a propagation scheme derived from **personalized PageRank**</u>
            - add a chance of **teleporting back to the root node** -> **PageRank** score <u>encodes the local neighborhood for every root node</u>
            - > the teleport vector allow us to preserve the node;s local neighborhood even in the limit distribution
        - show the propagation scheme permits the use of far more propagation steps **without lead to oversmoothing**
        - <u>The algorithm separates the neural network from the propagation scheme</u> -> achieve higher range without changing NN
            - > decouples prediction and propagation and solves the limited range problem inherent in many message passing models without introducing any additional parameters
        - **independent** development of the propagation algorithm and the nerual network generating predictions from node features
        - adding the propagation sheme during **inference** could significantly improves the accuracy without using any graph information
    - **<u>Other Notes</u>**:
        - Deep Learning on Graph: 
            - node embedding (without node features, normally unsupervised)
            - use graph structure and node features (supervised)
                - **[i]** spectral GCN
                - **[ii]** message passing
                - **[iii]** neighbor aggregation via RNN: *very limited neighborhood for each node*
            - <u>Increasing the size of the neighborhood</u> -> Laplacian smoothing and too many layers leed to **oversmoothing**
        - `solved by author in this paper`**[why GCN cannot be trivially expanded to use a larger neighborhood]**:
            - **[1]** aggregation by averaging causes oversmoothing if <u>too many layers</u> 
            - **[2]** Most use learnable weight matrices in each layer -> **increases <u>depth</u> and <u>number</u> of parameters in large neighborhood**
        - Differences between **limited distribution** & **PageRank**: 
            - added self-loops
            - adjacency matrix normalization

    - **<u>Use cases</u>**:
    - **<u>Further direction</u>**:
        - Combine PPNP with more complex neural networks used
        - faster or incremental approximations of personalized PageRank
        - More sophisticated propagation schemes 

    - **<u>Important references</u>**:
        - > Jiezhong Qiu, Yuxiao Dong, Hao Ma, Jian Li, Kuansan Wang, and Jie Tang. Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec. In ACM International Conference on Web Search and Data Mining (WSDM), 2018.

---


(21) `May 2019` [Approximation Ratios of Graph Neural Networks for Combinatorial Problems](https://arxiv.org/abs/1905.10261) *[Ryoma Sato, Makoto Yamada, Hisashi Kashima]*

* **Abstract**: 

> In this paper, from a theoretical perspective, we study how powerful graph neural networks (GNNs) can be for learning approximation algorithms for combinatorial problems. To this end, we first establish a new class of GNNs that can solve strictly a wider variety of problems than existing GNNs. Then, we bridge the gap between GNN theory and the theory of distributed local algorithms to theoretically demonstrate that the most powerful GNN can learn approximation algorithms for the minimum dominating set problem and the minimum vertex cover problem with some approximation ratios and that no GNN can perform better than with these ratios. This paper is the first to elucidate approximation ratios of GNNs for combinatorial problems. Furthermore, we prove that adding coloring or weak-coloring to each node feature improves these approximation ratios. This indicates that preprocessing and feature engineering theoretically strengthen model capabilities.

* **Key notes**: 


---

(22) `Apr 2019` [D-VAE: A Variational Autoencoder for Directed Acyclic Graphs](https://arxiv.org/abs/1904.11088) *[Muhan Zhang, Shali Jiang, Zhicheng Cui, Roman Garnett, Yixin Chen]*

* **Abstract**: 

> Graph structured data are abundant in the real world. Among different graph types, **directed acyclic graphs (DAGs)** are of particular interest to machine learning researchers, as many machine learning models are realized as computations on DAGs, including neural networks and Bayesian networks. In this paper, we study deep generative models for DAGs, and propose a novel DAG variational autoencoder (D-VAE). To encode DAGs into the latent space, we leverage graph neural networks. We propose an **[asynchronous message passing scheme]** that allows encoding the computations on DAGs, rather than using existing simultaneous message passing schemes to encode local graph structures. We demonstrate the effectiveness of our proposed D-VAE through two tasks: **neural architecture search** and **Bayesian network structure learning**. Experiments show that our model not only generates novel and valid DAGs, but also produces a **smooth** latent space that facilitates searching for DAGs with better performance through Bayesian optimization.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** propose **D-VAE**: <u>a variational autoencoder for DAGs</u> using a **novel asynchronous message passing scheme** 
            - able to **injectively encode computations** (*can build a mapping from the discrete space to a continuous latent space => every DAT computation has its unique embedding in the latent space*)
            - leverage GNN, **but** they ***simultaneously** passing all nodes' neighbors' messages to themselves* ==> designed ot learn **local structure features**, not suitable for DAGs because: 
                1. nodes are not symmetric, <u>intrinsically hae some ordering based on its dependency strucutre</u>
                2. more concerned about the computation represented by the **entire** graph, **not the local** strucutre
            - > Definition - **Computation**: composition of a finite number of operations applied to an input signal x, with the ouput of each operation being the input to its secceeding operations
            - **[Encoding]**: <u>graph neural network (GNN)</u> ==> <u>sequentially perform message passing for nodes following a topologica ordering of the DAG</u>
                - > **Theorem 1**: The D-VAE encoder is **invariant** to node permutations of the input DAT if teh aggregatoin functio is invariant to the order of its inputs
                - > **Theorem 2**: the encoder of D-VAE maps computation to hidden vector injectively if aggregator function is **injective** and update function is **injective**
                - <u>Themroem 2</u> provides a way to injectively encode computations on DAGs ==> every computation has a unique embedding in latent space 
            - **[Decoding]**: for new nodes: 
                1. predict its node type based on the current graph state
                2. sequentially predict whether each existing node has a directed edge to it based on the existing and current nodes' hidden states
        - **[2]** propose a new **DAG optimization framework** ==> <u>performs Bayesian optimization in a continuous latent space</u>
        - **[3]** apply D-VAE to two problems: `neural architecture search` and `Bayeisan network structure learning`
            - Evaluation schemes: 
                1. **[Accuracy]**: how often they can reconstruct input DAGs perfectly
                2. **[Validity]**: how often they can generate valid neural architectures or bayesian network from the prior distribution
                3. **[Uniqueness]**: the proportion of unique DAGs out of the valid generations
                4. **[Novelty]**: the proportion of valid generations that are never seen in the training set
    - <u>**Other Notes**</u>:
        - DAG optimization is very challenging: 
            1. the evaluation of the performance is time-consuming (training NN)
            2. SOTA optimization techiniques (**simulated annealing** & **Bayesian optimization (need kernal)**) operate in a **continuous** space ==> not directly applicable to DAG optimization due to the **discrete** nature
        - **[Deep graph generative models]**:   
            - `token-based`: <u>represent a graph as a sequence of tokens and model using RNNs</u>
            - `Adjacency-matrix-based models`: <u>leverage proxy adjacency matrix represnetation of a graph</u> ===> generate matrix in one shot or sequentially 
            - `graph-based`: operate **directly** on graph structures by <u>iteratively adding new nodes/edges to a graph based on existing graph and node states</u>
    - <u>**Use cases**</u>:
        - **[Neural architecture search (NAS)]**: searching optimal neural architectures (DAG optimization task)
            - <u>automating the design of neural network architectures</u>
            - `RL methods` (generate architectures with high rewards in term of validation accuracy)
            - `Bayesian optimization based method` (define kernels to measure architecture similarity and extrapolate the architecutre space heuristically)
            - `evolutionary approaches` (use evolutionary algorithms to optimize neural architectures)
            - `differentiable methods` (use continuous relaxation/mapping of neural architectures to enable gradient-based optimization)
        - **[Bayesian network structure learning (BNSL)]**: optimizing the connection structures of Bayesian networks
            - <u>learn the structure of the underlying bayesian network from observed data</u>
            - `score-based search` (define some "goodness-of-fit" score (*BIC and BDeu -> on marginal likelihood*) for network structure and search for one with optimal score in the discrete design space)
        - optimizing DAG circuit blocks ==> [realize target functions] && [meet specifications such as pwer usage & operating temperature]
    - <u>**Further directions**</u>:
        - currently use **one-hot** encoding of node types ==> could use pretrained embeddings of node types to replace it

---

(23) `May 2019` [End to end learning and optimization on graphs](https://arxiv.org/abs/1905.13732) *[Bryan Wilder, Eric Ewing, Bistra Dilkina, Milind Tambe]*

* **Abstract**: 

> Real-world applications often combine learning and optimization problems on graphs. For instance, our objective may be to cluster the graph in order to detect meaningful communities (or solve other common graph optimization problems such as facility location, maxcut, and so on). However, graphs or related attributes are often only partially observed, introducing learning problems such as link prediction which must be solved prior to optimization. We propose an approach to integrate a differentiable proxy for common graph optimization problems into training of machine learning models for tasks such as link prediction. This allows the model to focus specifically on the downstream task that its predictions will be used for. Experimental results show that our end-to-end system obtains better performance on example optimization tasks than can be obtained by combining state of the art link prediction methods with expert-designed graph optimization algorithms.

* **Key notes**: 
    - 

---

(24) `May 2019` [Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels](https://arxiv.org/abs/1905.13192) *[Simon S. Du, Kangcheng Hou, Barnabás Póczos, Ruslan Salakhutdinov, Ruosong Wang, Keyulu Xu]*

* **Abstract**: 

> While graph kernels (GKs) are easy to train and enjoy provable theoretical guarantees, their practical performances are limited by their expressive power, as the kernel function often depends on hand-crafted combinatorial features of graphs. Compared to graph kernels, graph neural networks (GNNs) usually achieve better practical performance, as GNNs use multi-layer architectures and non-linear activation functions to extract high-order information of graphs as features. However, due to the large number of hyper-parameters and the non-convex nature of the training procedure, GNNs are harder to train. Theoretical guarantees of GNNs are also not well-understood. Furthermore, the expressive power of GNNs scales with the number of parameters, and thus it is hard to exploit the full power of GNNs when computing resources are limited. The current paper presents a new class of graph kernels, Graph Neural Tangent Kernels (GNTKs), which correspond to \emph{infinitely wide} multi-layer GNNs trained by gradient descent. GNTKs enjoy the full expressive power of GNNs and inherit advantages of GKs. Theoretically, we show GNTKs provably learn a class of smooth functions on graphs. Empirically, we test GNTKs on graph classification datasets and show they achieve strong performance.

* **Key notes**: 
    - 

---

(25) `Sep 2018` [HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs](https://arxiv.org/abs/1809.02589) *[Naganand Yadati, Madhav Nimishakavi, Prateek Yadav, Vikram Nitin, Anand Louis, Partha Talukdar]*

* **Abstract**: 

> In many real-world network datasets such as co-authorship, co-citation, email communication, etc., relationships are complex and go beyond pairwise. Hypergraphs provide a flexible and natural modeling tool to model such complex relationships. The obvious existence of such complex relationships in many real-world networks naturaly motivates the problem of learning with hypergraphs. A popular learning paradigm is hypergraph-based semi-supervised learning (SSL) where the goal is to assign labels to initially unlabeled vertices in a hypergraph. Motivated by the fact that a graph convolutional network (GCN) has been effective for graph-based SSL, we propose HyperGCN, a novel GCN for SSL on attributed hypergraphs. Additionally, we show how HyperGCN can be used as a learning-based approach for combinatorial optimisation on NP-hard hypergraph problems. We demonstrate HyperGCN's effectiveness through detailed experimentation on real-world hypergraphs.

* **Key notes**: 
    - 

---

(26) `Jul 2019` [Social-BiGAT: Multimodal Trajectory Forecasting using Bicycle-GAN and Graph Attention Networks](https://arxiv.org/abs/1907.03395) *[Vineet Kosaraju, Amir Sadeghian, Roberto Martín-Martín, Ian Reid, S. Hamid Rezatofighi, Silvio Savarese]*

* **Abstract**: 

> Predicting the future trajectories of **multiple interacting agents** in a scene has become an increasingly important problem for many different applications ranging from control of autonomous vehicles and social robots to security and surveillance. This problem is compounded by the presence of social interactions between humans and their physical interactions with the scene. While the existing literature has explored some of these cues, they mainly ignored the **multimodal nature** of each human's future trajectory. In this paper, we present **Social-BiGAT**, a graph-based generative adversarial network that generates realistic, multimodal trajectory predictions by better modelling the social interactions of pedestrians in a scene. Our method is based on a **graph attention network (GAT)** that learns reliable feature representations that encode the social interactions between humans in the scene, and a **recurrent encoder-decoder architecture** that is trained adversarially to predict, based on the features, the humans' paths. We explicitly account for the multimodal nature of the prediction problem by forming a reversible transformation between each scene and its latent noise vector, as in Bicycle-GAN. We show that our framework achieves state-of-the-art performance comparing it to several baselines on existing trajectory forecasting benchmarks.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** propose **Social-BiGAT**, `GAN based approach to learn essential multimodel trajectory distributions`
            - input: `scene information` & `previously observed trajectory`
            - four main networks: 
                - **i** generator: 
                    1. `feature encoder module`(social pedestrian - **MLP** & physical scene - **CNN**) 
                    2. `attention network module` -> **Physical attention** & **Social attention** 
                    3. `decoder module` -> **GAN**
                - **ii** discriminator (*local pedestrain scale*)
                - **iii** discriminator (*global scene-level scale*)
                - **iv** latent space encoder -> *Multimodel*
                    1. Physical attention
                    2. Social Attention
        - **[2]** In troduce a fexible **graph attention network** -> <u>improve the modeling of social interactions between pedestrains</u>
            - **formulate pedestrian interactions as a graph**
        - **[3]** constructing a **reversible mapping** between <u>outputted trajectories</u> and <u>latents (represent the pedestrian behaviour)</u> -> encourage generalization towards a **multimodel** distribution
        - **[4]** incorporate physical scene cues using **soft attention** -> make the model more **generalizable**
    - <u>**Other Notes**</u>:
        -<u>Human trajectory prediction</u>: the problem of predicting the future navigation movements of pedestrians
             - given their `prior movement` and `additional contextual information`
        - trajectory prediction is still **challenging** due to <u>properties of human behaviour</u>:
            - **[i]** Social Interactions: (*require prediction methods to model social behaviour*)
            - **[ii]** Scene Context: (*people around the pedestrian*)
            - **[iii]** Multimodal behavior: (*human motion is inherently multimodal*)
        - previous work have limitation: 
            - not consider the physical cues of the scene
            - prior methods for trajectory prediction: <u>limit interactions to nearby pedestrian neighbors</u>
        - modern **Trajectory Forecasting** rely on **RNN**


    - <u>**Use cases**</u>:
        1. control of autonomous vehicles
        2. social robots to security and surveillance
        3. accurate pedestrian trajectory forecasting 
            - delivery vehicles -> understand human movement and avoid collisions
            - 
        4. Trajectory prediction for downstream tasks -> <u>tracking and reidentification</u>

    - <u>**Further directions**</u>:

---

(27) `May 2019 ` [Scalable Gromov-Wasserstein Learning for Graph Partitioning and Matching](https://arxiv.org/abs/1905.07645) *[Hongteng Xu, Dixin Luo, Lawrence Carin]*

* **Abstract**: 

> We propose a scalable Gromov-Wasserstein learning (S-GWL) method and establish a novel and theoretically-supported paradigm for large-scale graph analysis. The proposed method is based on the fact that Gromov-Wasserstein discrepancy is a pseudometric on graphs. Given two graphs, the optimal transport associated with their Gromov-Wasserstein discrepancy provides the correspondence between their nodes and achieves graph matching. When one of the graphs has isolated but self-connected nodes (i.e., a disconnected graph), the optimal transport indicates the clustering structure of the other graph and achieves graph partitioning. Using this concept, we extend our method to multi-graph partitioning and matching by learning a Gromov-Wasserstein barycenter graph for multiple observed graphs; the barycenter graph plays the role of the disconnected graph, and since it is learned, so is the clustering. Our method combines a recursive K-partition mechanism with a regularized proximal gradient algorithm, whose time complexity is $(K(E+V)logKV)$ for graphs with V nodes and E edges. To our knowledge, our method is the first attempt to make Gromov-Wasserstein discrepancy applicable to large-scale graph analysis and unify graph partitioning and matching into the same framework. It outperforms state-of-the-art graph partitioning and matching methods, achieving a trade-off between accuracy and efficiency.

* **Key notes**: 
    - 

---

(28) `May 2019` [Universal Invariant and Equivariant Graph Neural Networks](https://arxiv.org/abs/1905.04943) *[Nicolas Keriven, Gabriel Peyré]*

* **Abstract**: 

> Graph Neural Networks (GNN) come in many flavors, but should always be either invariant (permutation of the nodes of the input graph does not affect the output) or equivariant (permutation of the input permutes the output). In this paper, we consider a specific class of invariant and equivariant networks, for which we prove new universality theorems. More precisely, we consider networks with a single hidden layer, obtained by summing channels formed by applying an equivariant linear operator, a pointwise non-linearity and either an invariant or equivariant linear operator. Recently, Maron et al. (2019) showed that by allowing higher-order tensorization inside the network, universal invariant GNNs can be obtained. As a first contribution, we propose an alternative proof of this result, which relies on the Stone-Weierstrass theorem for algebra of real-valued functions. Our main contribution is then an extension of this result to the equivariant case, which appears in many practical applications but has been less studied from a theoretical point of view. The proof relies on a new generalized Stone-Weierstrass theorem for algebra of equivariant functions, which is of independent interest. Finally, unlike many previous settings that consider a fixed number of nodes, our results show that a GNN defined by a single set of parameters can approximate uniformly well a function defined on graphs of varying size.

* **Key notes**: 
    - 

---

(29) `Jun 2019` [Provably Powerful Graph Networks](https://arxiv.org/pdf/1905.11136.pdf) *[Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, Yaron Lipman]*

* **Abstract**: 

> Recently, the Weisfeiler-Lehman (WL) graph isomorphism test was used to measure the **expressive power** of graph neural networks (GNN). It was shown that the popular message passing GNN cannot distinguish between graphs that are **[indistinguishable]** by the 1-WL test (Morris et al. 2018; Xu et al. 2019). Unfortunately, many simple instances of graphs are indistinguishable by the 1-WL test.In search for more expressive graph learning models we build upon the recent k-order **invariant** and **equivariant** graph neural networks (Maron et al. 2019a,b) and present two results:

> **First**, we show that such **[k-order networks can distinguish between non-isomorphic graphs as good as the k-WL tests]**, which are provably stronger than the 1-WL test for k>2. This makes these models strictly stronger than message passing models. Unfortunately, the higher expressiveness of these models comes with a computational cost of processing high order tensors.

> **Second**, setting our goal at building a provably stronger, simple and scalable model we show that a **[reduced 2-order network containing just scaled identity operator, augmented with a *single quadratic operation* (matrix multiplication) has a provable 3-WL expressive power]**. Differently put, we suggest a simple model that interleaves applications of standard Multilayer-Perceptron (MLP) applied to the feature dimension and matrix multiplication. We validate this model by presenting state of the art results on popular graph classification and regression tasks. To the best of our knowledge, this is the first practical invariant/equivariant model with guaranteed 3-WL expressiveness, strictly stronger than message passing models.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - GOAL: <u>explore and develope GNN models that possess higher expressiveness while maintaining scalability</u>
        - MAIN CHALLENGE: <u>difficult to represent the multisets of neighborhoods required for the WL algorithms</u>
        - **[1]** establishing a baseline for expressive GNNs 
            - prove that recent **k-order invariant** GNN offer a natural hierarcht of models that are as expressive as the **k-WL** tests
        - **[2]** develope a simple model -> <u>incoporates standard **MLP** of the feature dimension and a **matrix multiplication layer**</u>
            - as far as they know, the model is the first to offer both **expressiveness (3-WL)** and **scalability (k=2)**
            - **Color representation**: *represent colors as vecotrs*
            - **Multiset representation**: encode multiset using a <u>set of Sn - invariant functions</u> -> **Power-sum Multi-symmetric Polynomials (PMP)**
    - <u>**Other Notes**</u>:
        - > **Message passing nerual networks**: node features are propagated through the graph according to its connectivity structure
        - previously, it is suggested to <u>compare the model's ability to distinguish between two given graphs to that of the hierarchy of the **Weisfeiler-Lehman (WL)** graph isomorphism tests</u>
        - Known results of **WL** and **FWL** algorithms: 
            1. **1-WL** and **2-WL** have equivalent discrimination power
            2. **k-FWL** is equivalent to **(k+1)-WL** for k>=2
            3. For each k>= 2 there is a pair of non-isomorphic graphs distinguishable by (k+1)-WL but not by k-WL
        - **Challenges** while analyzing networks ability to implement WL-like algorithms:
            1. Representing the coloar sigma in the nework
            2. implementing a multiset representation
            3. implementing the encoding function
    - <u>**Use cases**</u>:
        - Graphs are used to model -> `social network`, `chemical compounds`, `biological structures` and `high-level image content information`
    - <u>**Further directions**</u>: 
        1. search for more efficient GNN models with high expressiveness
        2. quantifying the generalization ability of these models

---

(30) `Jun 2017` [HARP: Hierarchical Representation Learning for Networks](https://arxiv.org/abs/1706.07845) *[Haochen Chen, Bryan Perozzi, Yifan Hu, Steven Skiena]*

* **Abstract**: 

> We present HARP, a novel method for learning low dimensional embeddings of a graph's nodes which preserves higher-order structural features. Our proposed method achieves this by compressing the input graph prior to embedding it, effectively avoiding troublesome embedding configurations (i.e. local minima) which can pose problems to non-convex optimization. HARP works by finding a smaller graph which approximates the global structure of its input. This simplified graph is used to learn a set of initial representations, which serve as good initializations for learning representations in the original, detailed graph. We inductively extend this idea, by decomposing a graph in a series of levels, and then embed the hierarchy of graphs from the coarsest one to the original graph. HARP is a general meta-strategy to improve all of the state-of-the-art neural algorithms for embedding graphs, including DeepWalk, LINE, and Node2vec. Indeed, we demonstrate that applying HARP's hierarchical paradigm yields improved implementations for all three of these methods, as evaluated on both classification tasks on real-world graphs such as DBLP, BlogCatalog, CiteSeer, and Arxiv, where we achieve a performance gain over the original implementations by up to 14% Macro F1.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(31) `Feb 2017` [Community Preserving Network Embedding](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14589) *[Xiao Wang, Peng Cui, Jing Wang, Jian Pei, Wenwu Zhu, Shiqiang Yang]*

* **Abstract**: 

> Network embedding, aiming to learn the low-dimensional representations of nodes in networks, is of paramount importance in many real applications. One basic requirement of network embedding is to preserve the structure and inherent properties of the networks. While previous network embedding methods primarily preserve the microscopic structure, such as the first- and second-order proximities of nodes, the mesoscopic community structure, which is one of the most prominent feature of networks, is largely ignored. In this paper, we propose a novel Modularized Nonnegative Matrix Factorization (M-NMF) model to incorporate the community structure into network embedding. We exploit the consensus relationship between the representations of nodes and community structure, and then jointly optimize NMF based representation learning model and modularity based community detection model in a unified framework, which enables the learned representations of nodes to preserve both of the microscopic and community structures. We also provide efficient updating rules to infer the parameters of our model, together with the correctness and convergence guarantees. Extensive experimental results on a variety of real-world networks show the superior performance of the proposed method over the state-of-the-arts.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(32) `Feb 2019` [Collaborative Similarity Embedding for Recommender Systems](https://arxiv.org/abs/1902.06188) *[Chih-Ming Chen, Chuan-Ju Wang, Ming-Feng Tsai, Yi-Hsuan Yang]*

* **Abstract**: 

> We present collaborative similarity embedding (CSE), a unified framework that exploits comprehensive collaborative relations available in a user-item bipartite graph for representation learning and recommendation. In the proposed framework, we differentiate two types of proximity relations: direct proximity and k-th order neighborhood proximity. While learning from the former exploits direct user-item associations observable from the graph, learning from the latter makes use of implicit associations such as user-user similarities and item-item similarities, which can provide valuable information especially when the graph is sparse. Moreover, for improving scalability and flexibility, we propose a sampling technique that is specifically designed to capture the two types of proximity relations. Extensive experiments on eight benchmark datasets show that CSE yields significantly better performance than state-of-the-art recommendation methods.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(33) `Apr 2019` [Graph Matching Networks for Learning the Similarity of Graph Structured Objects](https://arxiv.org/abs/1904.12787) *[Yujia Li, Chenjie Gu, Thomas Dullien, Oriol Vinyals, Pushmeet Kohli]*

* **Abstract**: 

> This paper addresses the challenging problem of **retrieval** and **matching** of *graph structured objects*, and makes two key contributions. 

> First, we demonstrate how Graph Neural Networks (GNN), which have emerged as an effective model for various supervised prediction problems defined on structured data, can **be trained to produce embedding of graphs in vector spaces that enables efficient similarity reasoning**. 

> Second, we propose a **novel Graph Matching Network model** that, given a pair of graphs as input, **computes a similarity score between them by jointly reasoning on the pair through a new cross-graph attention-based matching mechanism**. 

> We demonstrate the effectiveness of our models on different domains including the challenging problem of **control-flow-graph based function similarity search** that plays an important role in the detection of vulnerabilities in software systems. The experimental analysis demonstrates that our models are not only able to exploit structure in the context of similarity learning but they can also outperform domain-specific baseline systems that have been carefully hand-engineered for these problems.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** train **GNN** to produce <u>embedding of graphs</u> in vector spaces (*graph independently to vector*) -> further similarity computation happens in vector space
            - **(i) encoder**: node and edge features --MLPs--> initial vecotrs
            - **(ii) propagation layers**: set of node representations ----> new representations
                - *without propagation* -> *Deep set* or *PointNet* (<u>ignore the </u>)
            - **(iii) aggregator**: after T rounds of propagation, aggregate all the node representation to a **graph level representation** ----> 
                - > transorms node representations and then uses weighted sum with gating vectors to aggregate across nodes
            - for similarity use: `Euclidean`, `cosine`, `Hamming`

        - **[2]** propose **Graph Matching Network** (GMN) -> for similarity learning
            - compute a similarity score through a **cross-graph attention mechanism** -> `associate nodes across graphs and identify differences`, `use atten-based module`
            - more powerful than embedding model
            - compared to *[graph kernel approches]*, the authors' method based **similarity learning framework** learns the simlarity **end2end**
        - **[3]** evalute the model on three tasks: `syntehtic graph edit-distance learning task` (capture structural similarity only); real world tasks: `binary function similarity search` and `mesh retrieval`

        - **[Problem 1]**: <u>each cross-graph matching step requires computation of the full attention matrices</u> ----> **expensive for large graph**
        - **[Problem 2]**: the matching models operates on pairs, cannot directly be used for indexing and searching through large databases 
    - <u>**Other Notes**</u>:
        - graphs -> encoding **relational structures**
        - **Graph kernel**: kernels on graphs designed to capture the **graph similarity**, can be used in kernel method for `graph classifcation`
            - kernels based on **limited-sized** sub-strucutures
            - kernels based on **sub-tree** structures
            - Can ge formulated: 
                - > [i] computig the features vector for each graph (the kernel embedding)
                - > [ii] take inner product between vectors to compute kernel values
        - <u>distance metric learning</u>:
            - [early work] assumes data already lies in a vector space -> only a linear metric metrix is learned
            - [more recently] been combined in `face verification` (CNN to map similar images to similar vectors)
            - [this work] modeling cross-graph matchings
        - <u>Graph edit distance</u>: the minimum number of edit operations needed to transform G1 to G2 (*add/removes/substitute*)
        
    - <u>**Use cases**</u>:
        - <u>control-flow-graph based function similarity search</u> -> **detection of vulnerabilities in software systems**
            - <u>In the past</u>: use calssical graph theoretical matching algorithm
    - <u>**Further directions**</u>:
        - **[1]** improve the efficiency of the matching modesl
        - **[2]** study different matching architectures
        - **[3]** adapt GNN capacity to application domains

---

(34) `Jul 2016` [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) *[Aditya Grover, Jure Leskovec]*

* **Abstract**: 

> Prediction tasks over nodes and edges in networks require careful effort in engineering features used by learning algorithms. Recent research in the broader field of **representation learning** has led to significant progress in automating prediction by learning the features themselves. However, present feature learning approaches are not expressive enough to capture the diversity of connectivity patterns observed in networks. Here we propose node2vec, an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes. We define a flexible notion of a node's network neighborhood and design a biased random walk procedure, which efficiently explores diverse neighborhoods. Our algorithm generalizes prior work which is based on rigid notions of network neighborhoods, and we argue that the **added flexibility in exploring neighborhoods** is the key to learning richer representations. We demonstrate the efficacy of node2vec over existing state-of-the-art techniques on **multi-label classification** and **link prediction** in several real-world networks from diverse domains. Taken together, our work represents a new way for efficiently learning state-of-the-art task-independent representations in complex networks.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[i]** propose **node2vec** -> `efficient`, `scalable` algorithm for feature learning optimizes a `novel`, `network-aware`, `neighborhood preserving` objective using **SGD**
            - *use a 2nd order <u>random walk</u> approach to generate sample network neighborhoods for nodes*, (p,q) are use to represent the search bias
            - **Return parameter [p]**: <u>controls the likelihood of immedately revisiting a node in the walk</u>
            - **In-out parameter [q]**: <u>allows the search to differentiate between "inward" and "outward" nodes</u>:
                - if `q>1` -> biase toward close node -> BFS
                - if `q<1` -> bias towrad further nodes -> DFS
            - There is a implicit bias due to the start point: 
                - > **Solution**: simulating r random walks of fixed length l starting from every node
        - **[ii]** show node2vec in accordance with **principles in network science** -> [providing flexibility in discovering representations conforming to different equivalences]
            - > **homophily hypothesis**: nodes that are highly interconnected and belong to simliar network clusters or communities should be embedded closely together
            - > **structural equivalence hypothesis**: nodes have similar structural roles in networks should be embedded closely together
        - **[iii]** extend node2vec (and other feature learning methods) based on **neighborhood preserving objectives**, from node -> pair of nodes (edge prediction tasks)
        - **[iv]** evaluate node2vec on `multi-lable classification` and `link prediction`
    - <u>**Other Notes**</u>:
        - challenge in **feature representations**: <u>defining an objective function</u> [trade-off in balancing computational efficency and predictive accurcy]
        - essential points for a flexible algorithm to learn node representation
            - **[1]** ability to learn representatins that embed nodes from the same network community closely together
            - **[2]** learn representations where nodes that share similar roles have similar embeddingss
        - the conventional <u>dimensionality reduction techniques</u> have drawbacks in: 
            1. computational & statistical performance
            2. optimize for objectives that are **not robust** to the diverse pattern observed in networks *(assumptions -> relationship between underlying network structure and the prediction task)*
        - traditional search strategies: `Breadth-first sampling`, `Depth-first Sampling`
    - <u>**Use cases**</u>:
        1. predicting interests of users in a social network
        2. predicting functional labels of protein in protein-protein interatction network
        3. link prediction
            - discover novel interactions between genes in genomics
            - identify real-world friends in social network
    - <u>**Further directions**</u>:
        1. Explore reasons behind the success of Hadamard Operator over others (choice of binary operators for learning edge features)
        2. Node2Vec with special strucutre such as `heterogeneous information network`, `networks with explicit domain features for nodes and edges and signed-edge netowkr`

---

(35) `Jun 2019` [An Advanced Deep Generative Framework for Temporal Link Prediction in Dynamic Networks](https://ieeexplore.ieee.org/document/8736786) {local} *[Min Yang; Junhao Liu; Lei Chen; Zhou Zhao; Xiaojun Chen; Ying Shen]*

* **Abstract**: 

> Temporal link prediction in dynamic networks has attracted increasing attention recently due to its valuable real-world applications. The primary challenge of temporal link prediction is to capture the spatial-temporal patterns and high nonlinearity of dynamic networks. Inspired by the success of image generation, we convert the dynamic network into a sequence of static images and formulate the temporal link prediction as a conditional image generation problem. We propose a novel deep generative framework, called NetworkGAN, to tackle the challenging temporal link prediction task efficiently, which simultaneously models the spatial and temporal features in the dynamic networks via deep learning techniques. The proposed NetworkGAN inherits the advantages of the graph convolutional network (GCN), the temporal matrix factorization (TMF), the long short-term memory network (LSTM), and the generative adversarial network (GAN). Specifically, an attentive GCN is first designed to automatically learn the spatial features of dynamic networks. Second, we propose a TMF enhanced attentive LSTM (TMF-LSTM) to capture the temporal dependencies and evolutionary patterns of dynamic networks, which predicts the network snapshot at next timestamp based on the network snapshots observed at previous timestamps. Furthermore, we employ a GAN framework to further refine the performance of temporal link prediction by using a discriminative model to guide the training of the deep generative model (i.e., TMF-LSTM) in an adversarial process. To verify the effectiveness of the proposed model, we conduct extensive experiments on five real-world datasets. Experimental results demonstrate the significant advantages of NetworkGAN compared to other strong competitors.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(36) `Jan 2019` [GCN-GAN: A Non-linear Temporal Link Prediction Model for Weighted Dynamic Networks](https://arxiv.org/abs/1901.09165) *[Kai Lei, Meng Qin, Bo Bai, Gong Zhang, Min Yang]*

* **Abstract**: 

> In this paper, we generally formulate the dynamics prediction problem of various network systems (e.g., the prediction of mobility, traffic and topology) as the temporal link prediction task. Different from conventional techniques of temporal link prediction that ignore the potential non-linear characteristics and the informative link weights in the dynamic network, we introduce a novel non-linear model GCN-GAN to tackle the challenging temporal link prediction task of weighted dynamic networks. The proposed model leverages the benefits of the graph convolutional network (GCN), long short-term memory (LSTM) as well as the generative adversarial network (GAN). Thus, the dynamics, topology structure and evolutionary patterns of weighted dynamic networks can be fully exploited to improve the temporal link prediction performance. Concretely, we first utilize GCN to explore the local topological characteristics of each single snapshot and then employ LSTM to characterize the evolving features of the dynamic networks. Moreover, GAN is used to enhance the ability of the model to generate the next weighted network snapshot, which can effectively tackle the sparsity and the wide-value-range problem of edge weights in real-life dynamic networks. To verify the model's effectiveness, we conduct extensive experiments on four datasets of different network systems and application scenarios. The experimental results demonstrate that our model achieves impressive results compared to the state-of-the-art competitors.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(37) `May 2019` [Influential Nodes Detection in Dynamic Social Networks](https://link.springer.com/chapter/10.1007/978-3-030-20482-2_6) *[Nesrine HafieneEmail, authorWafa KarouiLotfi, Ben Romdhane]*

* **Abstract**: 

> The influence maximization problem aims to identify influential nodes allowing to reach the viral marketing objectives on social networks. Previous researches are mainly concerned with the static social network analysis and the development of algorithms in this context. However, when network changes, those algorithms must be updated. In this paper, we offer a new interesting approach to study the influential nodes detection problem in changing social networks. This approach can be considered to be an extension of a previous static algorithm SND (Semantic and structural influential Nodes Detection). Experimental results prove the effectiveness of SNDUpdate to detect influential nodes in dynamic social networks.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(38) `Jul 2019` [Factorization Bandits for Online Influence Maximization](https://arxiv.org/abs/1906.03737) *[Qingyun Wu, Zhige Li, Huazheng Wang, Wei Chen, Hongning Wang]*

* **Abstract**: 

> We study the problem of online influence maximization in social networks. In this problem, a learner aims to identify the set of "**best influencers**" in a network by interacting with it, i.e., *(repeatedly selecting seed nodes and observing activation feedback in the network)*. We capitalize on an important property of the influence maximization problem named **[network assortativity]**, which is ignored by most existing works in online influence maximization. To realize network assortativity, -> *(we factorize the activation probability on the edges into latent factors on the corresponding nodes)*, including influence factor on the giving nodes and susceptibility factor on the receiving nodes. We propose an **upper confidence bound** based online learning solution to estimate the latent factors, and therefore the activation probabilities. Considerable regret reduction is achieved by our **factorization based online influence maximization algorithm**. And extensive empirical evaluations on two real-world networks showed the effectiveness of our proposed solution.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** propose to model the **dependency** of **activation probabilities** on the edges for <u>online influence maximization</u> (*without the need of edge features*)
            - [Assumption 1]: each network nodes's <u>influence and susceptibility</u> are **distinct** and **individually specified**
            - [Assumption 2]: activation probability on an edge is jointly determined by <u>the **giving node's influence** and **receiving node's susceptibility**</u>
        - **[2]** propose a **factorization based bandit solution** -> to learn the <u>latent influence factors and susceptibility factors</u>
            - assuming <u>activation probability on edge e canbe decomposed into two d-dim latent factors on the giving node and receiving node</u> (**influence factor** + **susceptibility factor**)
            - assuming <u>the diffusion follows the **independent cascade model**</u>
            - introduce **L2-regularization** in the objective function: 
                1. *makes the sub-problem in coordinate decent based optimization **well-proposed*** ==> <u>closed form solution</u>
                2 *helps to remove the scaling indeterminacy between the estimates of the two factors*, and make the *q-linear convergence rate of parameter estimation achievable*
            - <u>Two definitions: </u>
                - > **Observed edge**: for any rount, a directed edge is considered as observed **iff** its start node is activated
                - > **Observed node**: A node is observed **iff** at least one of its giving-neighbor nodes is active
            - <u>Tree important advantages:</u>
                1. able to capture the **assortative mixing property** of influence distribution in a network
                2. activation observation from one edge can be readily leveraged to other edges <u>that share the same node</u>
                3. does not depend on the availability of manually constructed edge  or node level feautres (*it learns the property of network nodes via **factorization***)
            - <u>advanatages of **IMFB** to **CUCB**</u>:
                1. [reduced model complexity] 
                2. [reduced sampling complexity]
            - > in **CUCB**: *activation probability statistics* only be updated for those observed edges -> no information is learned for those unobserved edges
            - > in **IMFB**: utilizing the fact that **[the activation on a particular edge is a reflection of the giving node's influence]** and **[the receiving node's susceptibility]** ===> observations from the observed edges can be propagated to the unobserved edges
        - **[3]** provide rigorous theoretical analysis on the <u>upper regeret bound of the proposed solution</u> ==> prove considerable regret reduction comparint to solution that model the activation probability on edges independently. 
    - <u>**Other Notes**</u>:
        - online social networks play a vital role in the **spread of information ideas** and **influence among people**
        - **[influence maximization problem]**: *with a fixed budget on the number of selections, a marketer aims to maximize the spread of this influence*
            - **[activation probability]**: (Pe) represent the probability taht the reveiving node of edge e will be activated by the giving node on e
        - typically, in a social network:
            - it is associated with the **activation probility** -> <u>represent the connections or relationship between users</u>
            - influence is propagated through the network under a specific **diffusion model** (e.g. `independent cascade model`, `linear threshold model`)
        - Most existing influence maximization solutions assume the **activation probaibility is known beforehand** -> [however, this information might **not be observable** in many networks]
        - **[online influence maximization]**: <u>seed nodes are purposely selected by a learning agent</u> -> improve quality in influence estimation and influencer selection (based on **combinatorial bandits**)
            - *each node in the network as an arm*, at each round, the received reward on the selected set of seed nodes ==> the number of their activated nodes
        - Most existing [online influence maximization] model the activation probability **on the edges independently** -> <u>cannot capture how social influence forms in real networks</u>
        - referring to (independence/dependence) in <u>estimation of influence across network edges</u>: 
            1. **independence assumption** prevents the model from **realizign the deoendency among the influence patterns on related nodes** -> **[assortative mixing]**
                - > influential individuals are less susceptible to be influenced than non-influential individuals 
            2. the distribution of **influence** and **susceptibility** over network is ==> **heterogeneous**
                - to solve -> introduce **edge-level features** [however, many attributes are sensitive and hardly be exhausted, prohibited under privacy policy]
            3. edge-level estimzation of activation probability costs an algorithm high **computational complexity** and **sample complexity**
        - traditional **offline setting** -> <u>ignores many important properties (e.g. **assotativity**) of real network influence patterns</u>
        - [*In practive it is very difficult to exhaustively specify the features for influence modeling on every edge*]
        - previous works donnot explicitly separate influence and susceptibility of nodes ==> causes seriously degraded estimation of **influence propagation**
        - **[Online influence maximization with bandit]**: 
            - Traditional works assums: `knows the per-edge activation probability`, `the probability can be specified from past propagation data`
            - most existing badit-based OIM: `the learner estimates the activation probabilities on all edges independently` ==> <u>cannot capture how tht influence between nodes is ofrmed and ignores the underlying structure behind the activation probabilities of edges</u>
            - In each round of OIM, the learner needs to choose seeds to:
                1. <u>maximize influence spread</u> (**exploitation**)
                2. <u>improve its knowledge of the activation probabilities via feedback</u> (**exploration**)
    - <u>**Use cases**</u>:
        - marketer tries to select a set of customers with great influence for a new product promotion
        - datasets: `NetHEPT` and `Flickr`
    - <u>**Further directions**</u>:
        - incorporate more sophisticated inference mehtods based on the factor models to learn from **[node-level feedback is more prevalent instead of edge-level feedback (we assumed)]**

---

(39) `May 2018` [Semi-supervised User Geolocation via Graph Convolutional Networks](https://arxiv.org/abs/1804.08049) *[Afshin Rahimi, Trevor Cohn, Timothy Baldwin]*

* **Abstract**: 

> Social media user geolocation is vital to many applications such as **event detection**. In this paper, we propose GCN, a multiview geolocation model based on **Graph Convolutional Networks**, that uses **[both text and network context]**. We compare GCN to the state-of-the-art, and to two baselines we propose, and show that our model achieves or is competitive with the state- of-the-art over three benchmark geolocation datasets when sufficient supervision is available. We also evaluate GCN under a minimal supervision scenario, and show it outperforms baselines. We find that **[highway network gates are essential for controlling the amount of useful neighbourhood expansion in GCN]**.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** propose GCN, an **end-to-end user geolocation model based on <u>Graph Convolutional Networks</u>** -> jointly learns from **text** and **network** information
        - **[2]** evaluate the model under a <u>minimal supervision scenario</u> -> close to real world applications, show GCN outperforms than two strong baselines
            - `MLP-TXT+NET` based on **concatenation of text and network**
            - `DCCA` based on **Deep Canonical Correlation Analysis**
        - **[3]** given sufficient supervision -> GCN is competitive
        - **[4]** show highway gates play a significant role in controlling the amount of useful neighbourhood smoothing in GCN
    - <u>**Other Notes**</u>:
        - **[User geolocation]**: <u>identifying the "home" location of a user</u>
        - Social media **[Twitter]**: rely on `IP addresses`, `WiFi footprints`, `GPS data` to geolocate users
        - Most previous work on user geolocation uses: 
            - `supervised text-based approaches` -> rely on **geographical variation of language use**
            - `graph-based semi-supervised label propagation` -> rely on **location homophily in user-user interactions**
            - 
    - <u>**Use cases**</u>:
        - Event detection by socia media user geolocation
        - public health monitoring
        - regional studies of sentiment
        - real-time emergency awareness systems
    - <u>**Further directions**</u>:
        - modelling the extent to which a social interaction is caused by geographical proximity (using user-user gates)

---

(40) `Jun 2019` [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829) *[Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon]*

* **Abstract**: 

> Point clouds provide a flexible geometric representation suitable for countless applications in computer graphics; they also comprise the raw output of most 3D data acquisition devices. While hand-designed features on point clouds have long been proposed in graphics and vision, however, the recent overwhelming success of convolutional neural networks (CNNs) for image analysis suggests the value of adapting insight from CNN to the point cloud world. Point clouds inherently lack topological information so designing a model to recover topology can enrich the representation power of point clouds. To this end, we propose a new neural network module dubbed EdgeConv suitable for CNN-based high-level tasks on point clouds including classification and segmentation. EdgeConv acts on graphs dynamically computed in each layer of the network. It is differentiable and can be plugged into existing architectures. Compared to existing modules operating in extrinsic space or treating each point independently, EdgeConv has several appealing properties: It incorporates local neighborhood information; it can be stacked applied to learn global shape properties; and in multi-layer systems affinity in feature space captures semantic characteristics over potentially long distances in the original embedding. We show the performance of our model on standard benchmarks including ModelNet40, ShapeNetPart, and S3DIS.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(41) `Aug 2018` [Attributed Network Embedding for Learning in a Dynamic Environment](https://arxiv.org/pdf/1706.01860.pdf) *[Jundong Li, Harsh Dani, Xia Hu, Jiliang Tang, Yi Chang, Huan Liu]*

* **Abstract**: 

> Network embedding leverages the node proximity manifested to learn a low-dimensional **node vector representation** for each node in the network. The learned embeddings could advance various learning tasks such as **[node classification]**, **[network clustering]**, and **[link prediction]**. Most, if not all, of the existing works, are overwhelmingly performed in the context of plain and static networks. Nonetheless, in reality, network structure often **evolves over time with addition/deletion of links and nodes**. Also, a vast majority of real-world networks are associated with a rich set of node attributes, and their attribute values are also naturally changing, with the emerging of new content patterns and the fading of old content patterns. These changing characteristics motivate us to seek an effective embedding representation to capture network and attribute evolving patterns, which is of fundamental importance for learning in a dynamic environment. To our best knowledge, we are the first to tackle this problem with the following two challenges: 

> **(1)** the inherently correlated network and node attributes could be noisy and incomplete, it necessitates a robust consensus representation to capture their individual properties and correlations; 

> **(2)** the embedding learning needs to be performed in an online fashion to adapt to the changes accordingly. 

> In this paper, we tackle this problem by proposing a novel dynamic attributed network embedding framework - **DANE**. In particular, DANE first provides an **offline** method for a consensus embedding and then leverages **matrix perturbation theory** to maintain the freshness of the end embedding results in an online manner. We perform extensive experiments on both synthetic and real attributed networks to corroborate the effectiveness and efficiency of the proposed framework.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** **Problem Formulations**: formally define the problem of **[dynamic attributed netowrk embedding]**: <u>initiate an offline model at the very beginning, based on which an online model is presented to maintain the freshness of the end attributed network embedding results</u>
        - **[2]** **ALgorithms and Analysis**: propose a novel framework **[DANE]** for <u>dynamic attributed nework embedding</u>
            - introuce an **offline embedding method** as a based model -> preserve **node proximity interms of both [network structure] and [node attributes]** for consensus embedding representation in a **robust** way
            - workflow of proposed model DANE: 
                - **Problem 1**: <u>The offline model of DANE at time t</u> - given network topology At and node attributes Xt; output attributed netowrk embedding Yt for all nodes
                - **Problem 2**: <u>The online model of DANE at time step t+1</u> - given network topology A(t+1) and node attributes X(t+1) and intermediate embedding results at time step t -> output attributed network embedding Y(t+1) for all nodes
            - **Nodes**: network embedding could be **jeopardized** as <u>links are inadequate to provide enough node proximity information</u>
            - The proposed online embedding model is motivated by the **observation**: 
                - > most of the real-world networks, with no exception for attributed networks, often evolve smoothly in the temporal dimension between two consecutive time steps
            - The **core idea** to enable online update of the embeddings is --> <u>develop an efficient way to update the top eigenvectors and eigenvalues</u>
        - **[3]** **Evaluations**: perform extensive experiments on both **synthetic** and **real-world** attributed network --> <u>corroborate the efficacy in terms of two network mining tasks</u> (both `unsupervised` and `supervised`)
            - show efficiency by comparing to the baselines, results show that DANE outperforms the best competitors in terms of both <u>clustering</u> and <u>classfication performace</u>, and it's much **faster**
            - Attempts to answers two questions in the experiments: 
                1. **Effectiveness**: <u>how effective are the embeddings obtained by DANE on different learning tasks</u>
                2. **Efficiency**: how fast is the proposed framewrok DANE compared with other offline embedding methods
            - datasets: `BlogCatalog`, `Flickr`, `Epinions`, `DBLP`

    - <u>**Other Notes**</u>:
        - there exhibits a **[strong correlation among the attributes of linked nodes]**, with root causes: `social influence` and `homophily effect`
        - <u>Basic idea of **network embedding**</u>: preserve the node proximity in the embedded Euclidean space 
        - **fundamental assumption** behind existing network embedding methods: <u>networks are **static** and given a prior</u> 
        - <u>most real-world networks are intrinsically dynamic with **addition/deletion** of edges and nodes</u>: (examples: `co-author relations b/w scholar`, `friendships among users in a social network`)
        - Natural question to ask for dynamic attriguted networks: 
            - <u>when attributed netowrks evolve</u>
            - <u>how to correct and adjust the saleness of the end embedding results for network analysis</u>
        - dynamic attributed netowrk embedding is a **dauning task** because: 
            1. even though **network topology** and **node attributes** are two distinct data representations -> they are **[inherently correlated]** 
            2. *the raw data representations could be **noisy** and even **incomplete*** -> it is of **paramount importance** to seek a **noise-resilient consensus embedding** to capture their **individual properties** and **correlations**
            3. applying offline embedding methods from scratch at each time step is **time-consuming** and cannot seize the emerging patterns timely --> require an **efficient online algorithm**
        - In <u>attributed networks</u>, the **network structure** is <u>more than a single view of data as it encodes other types of rich information</u> --> such as **[connectivity]**, **[transitivity]**, **[reciprocity]**

    - <u>**Use cases**</u>:
        - **attributed networks** --> `social media networks`, `academic networks`, `protein-protein intereaction networks`
        - in **scientific collabration networks**: researchers collaborate and are distinct from others by their unique research interests
        - in **social networks**: users interact and communicate with others and also post personalized content
        - `node classifcation`, `community detection`, `topic modeling` and `anomaly detection`

    - <u>**Further directions**</u>:
        1. investigate how the **high-order** approximations can be applied to the online embedding learning problem (*first-order matrix matrix perturbation theory is employed in this paper*)
        2. extend the current framework to multi-mode and multi-dim dynamic network

---

(42) `Jun 2019` [Dynamic Network Embedding via Incremental Skip-gram with Negative Sampling](https://arxiv.org/abs/1906.03586) *[Hao Peng, Jianxin Li, Hao Yan, Qiran Gong, Senzhang Wang, Lin Liu, Lihong Wang, Xiang Ren]*

* **Abstract**: 

> **Network representation learning**, as an approach to learn low dimensional representations of vertices, has attracted considerable research attention recently. It has been proven extremely useful in many machine learning tasks over large graph. Most existing methods focus on learning the **structural representations** of vertices in a static network, but cannot guarantee an accurate and efficient embedding in a dynamic network scenario. The fundamental problem of **[continuously capturing the dynamic properties in an efficient way for a dynamic network]** remains unsolved. To address this issue, we present an **efficient incremental skip-gram algorithm with negative sampling for dynamic network embedding**, and provide a set of theoretical analyses to characterize the performance guarantee. Specifically, 

> **[1]** we first partition a dynamic network into the updated, including addition/deletion of links and vertices, and the retained networks over time. 

> **[2]** Then we factorize the objective function of network embedding into the added, vanished and retained parts of the network. 

> **[3]** Next we provide a new stochastic gradient-based method, guided by the partitions of the network, to update the nodes and the parameter vectors. 

> The proposed algorithm is proven to yield an objective function value with a bounded difference to that of the original objective function. Experimental results show that our proposal can significantly **reduce the training time** while **preserving the comparable performance**. We also demonstrate the correctness of the theoretical analysis and the practical usefulness of the dynamic network embedding. We perform extensive experiments on multiple real-world large network datasets over multi-label classification and link prediction tasks to evaluate the effectiveness and efficiency of the proposed framework, and up to 22 times speedup has been achieved.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
        - **[1]** A **dynmaic network embedding framework** based on an **approximately optimal solution** of <u>incremental skip-gram with negative sampling</u> is proposed ==> can be directly applied in existing network embedding models such as `DeepWalk` and `Node2Vec`
            - To address the issue of updating `structure proximities` and `noise distribution`: 
                1. partition the network in to the **updated part** and **retained part**
                2. employ **random walk** and **sliding window** to <u>extract the sequences of the nodes or subgraphs</u> -> **[affected sequences of subgraphs in the network]**
                3. the model <u>inherits all the retained nodes and parameter vectors and implements a new **stochastic gradient-based model** to update the changed nodes and parameter vectors</u>
            - Therefore, we only need to update vectors in affected subgraph
            - **Assumption**: the influence of <u>structural changes</u> on representation learning is partial in **neighborhoods/sub-graphs** for limited adjacency matrix float: 
                1. <u>locate the metabolic sub-graphs by **local random walk** in retraining</u>
                2. **For newly added nodes or edges**: <u>implement the random walk only on the sub-graphs to generate sequences of vertices</u>
                3. **For the vanished nodes or edges**: <u>implement the random walk on the sub-graphs to generate sequences of vertices following the same rule</u>
                4. re-count the frequency for each vertex appearing in the above sequences, and add/subtract its into/from the frequencies generated in the old network ===> <u>get the latest noise distribution</u>
            - For **fast training** in dynamic network scenario, adopt a strategy: 
                - > inherits the vertexes and the parameter vectors through changes in the netwrok structures
        - **[2]** solid theoretical analyses show that: the proposal **guarantees the boundness of the objective difference and the convergence when the training network scale is infinitely large** (*empirial study also verifies the boundary and moments of the network dynmaic change*)
        - **[3]** Experiments on <u>multiple large real-world network datasets show both the **efficiency** and **effectiveness** of the proposed **ISGNS** on **multi-label classfication** and **link prediction tasks**</u>
    - <u>**Other Notes**</u>:
        - The idea of **nework embedding**: <u>learn a mapping that projects each vertex in a network to a low dimensional and continuous distributed vector space</u>
        - **DeepWalk** & **node2vec**: *capture high-order proximities in embeddings by maximizing the conditional probability of observing the neighbourhood of vertices of a vertex given the mapped point of the vertex*
            - **Difference**: **node2vec** --> employs a biased random walk to provide a trade-off b/w **breadth-first search** and **depth-first search** in a network ==> better representation
        - **LINE** & **SDNE**: *preserving the first- and second-order proximities in the embedded space* (former refers to pairwise neighborhood relatoinship, and determined by the similarity of nodes' neighbors)
            - **Difference**: **SDNE** uses <u>highly non-linear function to represent the mapping function</u>
        - **Problem** of using `negative sampling` or `hierarchical softmax optimizing`: <u>when the difference b/w the updated network and the old network is **relatively small**</u> --> it's **inefficient** to obtain the new node embeddings through retraining the entire new network
        - **[Hierarchical softmax]**: <u>a hierarchical tree is constructed to index all the words in a corpus as leaves</u>
        - **[Negative sampling]**: <u>developed based on **noise constrastive estimzatio** and **randomly samples the words not in the context** to distinguish the observed data from the artificially generated random noise</u> 
        - **Problems** to apply **skip-gram with negative sampling** to network representation learning: [investigate the **structure proximities** and compute the **noise distributions** for negative sampling]: 
            - <u>when the vertices and edges of a network evolve over time, the proximities and noise distributions will update automatically to reflect the change of the network structure</u>
            - **In the dynamic scewnario**: when the `edges`, the `edge weights` and the `vertices` changes --> the **sequences of vertices**, the **structure proximities** and the **noise distribution** should be updated correspondingly
        - **[Static network embedding]**: 
            - `DeepWalk`: the first work that utilizes a **truncated random walk** to transform a static network into a collection of node sequences -> the <u>skip-gram on **hierarchial softmax** funciton is used</u>
            - `Node2vec`: further generalizes `DeepWalk` with **Breadth-First Search (BFS)** and **Depth-First Searc (DFS)** on random walks ==> employ <u>skip-gram with **negative sampling**</u>
            - `LINE` and `SDNE`: model the <u>first-order and second-order</u> proximities b/w **vertices** ==> employ <u>skip-gram with **negative sampling**</u> to deal with the limitatio of <u>stochastic gradient descent</u> on weighted edges **without compromising efficiency**
            - `Struct2vec`: propose to <u>preserve the structural identity b/w **nodes** in the representation</u> -->: 
                1. creates a new graph based on the <u>structural identity similarity b/w nodes</u>
                2. follows a similar method to `DeepWalk` on the created graph
            - `Graph-Wave`: use <u>wavelet diffusion patterns</u> by treating the wavelets from the **heat wavelet diffusion process** as distribution
        - **[Dynamic network embeding]**:
            - `DANE`: leverages a **matrix perturbation theory** --> to update the <u>dynamic attributed network spectral embeddings</u>
            - `BCGD`: <u>temporal latent space learning model</u> via **non-negative matrix factorization** to target the link prediction task in dynamic social networks
            - `Know-Evolve`: **deep recurrent architecture** ==> model the historical evolution of entity representations in a specific relationship space
            - `Dynamic SBM`: statistical model for dynamic networks that utilized a set of **unobserved time-varying states** ==> characterize the dynamics of the network
            - `OLSN`: online embedding representation learning method <u>based on spectral embedding</u> used for node classification
            - `Dynamic Triad`: <u>triadic closure process based semi-supervised algorithm</u> ==> to learn the structural oinformation and evolution pattern in dynamic network
            - `DNE`: <u>heuristic dynamic network embedding method</u> ==> employs a decomposable objective based on the skip-gram objective
            - `HTNE`: <u>Hawkes process based temporal network embedding method</u> ==> capture the influence of the historical neighbors on the current neighbor formation simultaneously

        - <u>network representation learning</u> is sensitive to the **[network structure]** and the **[objective proximities among vertices]**

    - <u>**Use cases**</u>:
        - `recommended systems`, `social networks`, `biology networks`
        - `vertext community detection`, `recommended system`, `anamoly detection`, `multi-label classifcation`, `link prediction`, `knowledge representation`
    - <u>**Further directions**</u>:
        - extend the approach to other advanced network representation learning models

---

(43) `August 2019` [TIFIM: A Two-stage Iterative Framework for Influence Maximization in Social Networks](https://www.sciencedirect.com/science/article/abs/pii/S0096300319301602) *[Qiang He, Xingwei Wang, Zhencheng Lei, Min Huang, Yuliang Cai, Lianbo Ma]*

* **Abstract**: 

> **Influence Maximization** is an important problem in social networks, and its main goal is to select some most influential initial nodes (i.e. **seed nodes**) to obtain the maximal influence spread. The existing studies primarily concentrate on the corresponding methods for influence maximization, including **[greedy algorithms]**, **[heuristic algorithm]** and their extensions to determine the most influential nodes. However, there is little work to ensure `efficiency` and `accuracy` of the proposed schemes at the same time. In this paper, a **Two-stage Iterative Framework for the Influence Maximization** in social networks, (i.e., TIFIM) is proposed. In order to `exclude less influential nodes` and `decrease the computation complexity of TIFIM`: 

> **[1]** In the first stage, an **iterative** framework in descending order is proposed to select the candidate nodes. In particular, based on the results of the last iteration and the two-hop measure, the **First-Last Allocating Strategy (FLAS)** is presented to compute the spread benefit of each node. We prove that TIFIM converges to a stable order within the finite iterations. 

> **[2]** In the second stage, we define the apical dominance to calculate the overlapping phenomenon of spread benefit among nodes and further propose Removal of the Apical Dominance (RAD) to determine seed nodes from the candidate nodes. 

> **[3]** Moreover, we also prove that the influence spread of TIFIM according to RAD converges to a specific value within finite computations. 

> **[4]** Finally, simulation results show that the proposed scheme has superior influence spread and running time than other existing ones.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(44) `Dec 2019` [Principled Weight Initialization for Hypernetworks ](https://openreview.net/forum?id=H1lma24tPB) *[Oscar Chang, Lampros Flokas, Hod Lipson]*

* **Abstract**: 

> Hypernetworks are meta neural networks that generate weights for a main neural network in an end-to-end differentiable manner. Despite extensive applications ranging from multi-task learning to Bayesian deep learning, the problem of optimizing hypernetworks has not been studied to date. We observe that classical weight initialization methods like Glorot & Bengio (2010) and He et al. (2015), when applied directly on a hypernet, fail to produce weights for the mainnet in the correct scale. We develop principled techniques for weight initialization in hypernets, and show that they lead to more stable mainnet weights, lower training loss, and faster convergence.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(45) `Nov 2019` [Contrastive Learning of Structured World Models](https://arxiv.org/abs/1911.12247) *[Thomas Kipf, Elise van der Pol, Max Welling]*

* **Abstract**: 

> A structured understanding of our world in terms of objects, relations, and hierarchies is an important component of human cognition. Learning such a structured world model from raw sensory data remains a challenge. As a step towards this goal, we introduce Contrastively-trained Structured World Models (C-SWMs). C-SWMs utilize a contrastive approach for representation learning in environments with compositional structure. We structure each state embedding as a set of object representations and their relations, modeled by a graph neural network. This allows objects to be discovered from raw pixel observations without direct supervision as part of the learning process. We evaluate C-SWMs on compositional environments involving multiple interacting objects that can be manipulated independently by an agent, simple Atari games, and a multi-object physics simulation. Our experiments demonstrate that C-SWMs can overcome limitations of models based on pixel reconstruction and outperform typical representatives of this model class in highly structured environments, while learning interpretable object-based representations.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(46) `Jun 2019` [Causal Discovery with Reinforcement Learning](https://arxiv.org/abs/1906.04477) *[Shengyu Zhu, Zhitang Chen]*

* **Abstract**: 

> Discovering causal structure among a set of variables is a fundamental problem in many empirical sciences. Traditional score-based casual discovery methods rely on various local heuristics to search for a directly acyclic graph (DAG) according to a predefined score function. While these methods, e.g., greedy equivalence search (GES), may have attractive results with infinite samples and certain model assumptions, they are less satisfactory in practice due to finite data and possible violation of assumptions. Motivated by recent advances in neural combinatorial optimization, we propose to use reinforcement learning (RL) to search for the DAG with the best scoring. Our encoder-decoder model takes observable data as input and generates graph adjacency matrices that are used to compute corresponding rewards. The reward incorporates both the predefined score function and two penalty terms for enforcing acyclicity. In contrast with typical RL applications where the goal is to learn a policy, we use RL as a search strategy and our final output would be the graph, among all graphs generated during training, that achieves the best reward. We conduct experiments on both synthetic and real data, and show that the proposed approach not only has an improved search ability but also allows for a flexible score function under the acyclicity constraint.

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(46) `201` []() *[]*

* **Abstract**: 

> 

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(46) `201` []() *[]*

* **Abstract**: 

> 

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(46) `201` []() *[]*

* **Abstract**: 

> 

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(46) `201` []() *[]*

* **Abstract**: 

> 

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(46) `201` []() *[]*

* **Abstract**: 

> 

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(46) `201` []() *[]*

* **Abstract**: 

> 

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

(46) `201` []() *[]*

* **Abstract**: 

> 

* **Key notes**: 
    - <u>**Main contributions**</u>: 
    - <u>**Other Notes**</u>:
    - <u>**Use cases**</u>:
    - <u>**Further directions**</u>:

---

**TO ADD**:

* [ ] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon. Dynamic graph cnn for learning on point clouds. arXiv preprint arXiv:1801.07829, 2018

---

### Extraction from paper

<u>**General Network**</u>

* > Network can effectively characterize many complex systems, where each node indicates an entity and each edge represents an iteraction between a pair of vertices *[An Advanced Deep Generative Framework for Temporal Link Prediction in Dynamic Networks]*

<u>**Social network**</u>

* > **source like twitter**: social interactions, sentiment analysis, content diffusion, link prediction, and **the dynamics behind human collective behaviour** in general

* > social networks, such as Facebook, Twitter and Wechat, have connected all web users in the Internet and received great attention from all over the world [1,2]. [1] H. Zhang, S. Mishra, M.T. Thai, Recent advances in information diffusion and influence maximization in complex social networks, Opportun. Mob. Soc. Netw. (2014) 1–37. [2] W. Tan, M.B. Blake, I. Saleh, S. Dustdar, Social-network-sourced big data analytics, IEEE Int. Comput. 17 (5) (2013) 62–69.

* > many company will promote their products on social network like instagram and twitter. Job provider might spread opportunities through social network. 

* > In social networks, the links are usually varying dynamically according to the behaviors of individual's socail patterns *ref[D. Jin, X. Wang, R. He, D. He, J. Dang, and W. Zhang, “Robust detec- tion of link communities in large social networks by exploiting link semantics,” in Proc. 32nd AAAI Conf. Artif. Intell., 2018, pp. 314–321.]*


<u>**Dynamic network**</u>

* > the dynamics of communication links in ad hoc networks makes the design of routing protocol a challenging problem


<u>**Link prediction**</u>

<u>**network embedding**</u>


<u>**influence maximization**</u>



---

### SNA/RP paper to read

**<u>Dynamic, embedding Link prediction</u>**

* [x] **(dynamic/embedding)**[Dynamic Network Embedding by Modeling Triadic Closure Process](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16572)
    - preserve not only the structure information but also the **evolotion pattern** over time
    - tasks: link prediction at the next time t+1
    - dynamic mainly for the link construction and link disolution

* [x] **(important)**(dynamic/embedding)[Attributed Network Embedding for Learning in a Dynamic Environment](https://arxiv.org/pdf/1706.01860.pdf)
    - we present an online model to update the consensus embedding with matrix perturbation theory

* [x] **(very new, with ref to all of the previous dynamic embedding method)**(Dynamic network) [Dynamic Network Embedding via Incremental Skip-gram with Negative Sampling](https://arxiv.org/abs/1906.03586)
    - the **network structur** is dynamic
        - > the edges, the edge weights and the vertices change, the sequences of vertices, the structure proximities and the noise distribution should be updated correspondingly
* [x] **(mentioned in the above)**(Dynamic network) [Dynamic Network Embedding: An Extended Approach for Skip-gram based Network Embedding](https://www.ijcai.org/proceedings/2018/0288.pdf)
* [x] (important, combine attribute & structure)(Embedding/link prediction)[Attributed Social Network Embedding](https://ieeexplore.ieee.org/abstract/document/8326519)
    -  combine Structural Proximity & Attribute Proximity by **early fusion** - optimize them simultaneously
    - Future direction at its time: 
        - **[1]** fusing data from multiple modalities (multi-modal data, image...)
        - **[2]** semi-supervised for task-oriented embedding
        - **[3]** capture evolution nature of social network (new users, new social relationship) -> temporal aware neural network
        - **[4]** improve efficiency
* [x] (Dynamic/Link prediction) [A Supervised Learning Approach to Link Prediction in Dynamic Networks](https://link.springer.com/chapter/10.1007/978-3-319-94268-1_70)
    - flat SVM for prob of link connecton

* [x] (Dynamic/Link Prediction)[Semi–supervised Graph Embedding Approach to Dynamic Link Prediction](https://arxiv.org/pdf/1610.04351.pdf)
    - semi-supervised, only consider the link modification (formation and dissoloution) between each of the two cconsecutive discrete time steps.
    - use all of the history information 1-t to predict the one at t+1
    - semi: supervised in history information + unsupervised in the semisupervised graph embedding

* [x] **(May 2018)(Dynamic/Heterogeneous)(fresh for forecasting the relation building time, dynamic in structure)** [Continuous-Time Relationship Prediction in Dynamic Heterogeneous Information Networks](https://arxiv.org/pdf/1710.00818.pdf)
    - solve the problem of continous-time relationship prediction in **dynamic** and **heterogeneous** information networks
    - propose `continuous-time link prediction problem`, <u>predict when a link will emerge or appear between two nodes in the network</u>
    - all nodes and links are associated with a birth and death time
    - combine LSTM and auto-encoder
    - Further direction:
        1. design unified architecture to combine feature extraction step with the learing algorithm in an integrated Deep Learning framework
        2. investigate node embedding and approximation techniques

* [x] **(Feb 2019)(Dynamic/link prediction)(dynamic in link formation and dissolusion)** [E-LSTM-D: A Deep Learning Framework for Dynamic Network Link Prediction](https://arxiv.org/pdf/1902.08329.pdf)
    - Encoder-LSTM-Decoder to predict dynamic links **end2end**
    - embedding still lack the ability of analyzing the evolution of network
    - define new metrix **[Error Rate]** to measure the performance of dynamic network link prediction
        - > Ratio of the number of mispredicted links to the total number of truly existing links
    - experienment on five real-world human contact dynamic network

* [x] **(Jan 2019)(Dynamic/link prediction)(Good to touch predict topology/traffic prediction)** [GCN-GAN: A Non-linear Temporal Link Prediction Model for Weighted Dynamic Networks](https://arxiv.org/pdf/1901.09165.pdf)
    - generally formulate the dynamic prediction problem of various network system (the prediction of `mobility`, `traffic`, `topology`)
    - GCN-LSTM-GAN
    - temporal link prediction task tries to construct the graph topology in the next time slice
    - Most of the link prediction only for unweighted network
    - first to use GAN to tackle the temporal link prediction of **weighted** dynamic network

* (x) **(Time Series/Link prediction/Heterogeneous)** [Multivariate Time Series Link Prediction for Evolving Heterogeneous Network](https://ideas.repec.org/a/wsi/ijitdm/v18y2019i01ns0219622018500530.html)
* [x] **(Temporal/link prediction/dynamic)(very good, with GCN, LSTM, GAN, have 4 datasets) [An Advanced Deep Generative Framework for Temporal Link Prediction in Dynamic Networks](https://ieeexplore.ieee.org/document/8736786) {local}
    - inherits `graph convolusional NN`, `temporal matrix factorization`(**TMF**), `LSTM` and `GAN`
    - propose TMF enhanced attentive LSTM (TMF-LSTM)
    - self-attention GCN
    - **TMF** for the global latent features and evolving pattern of the dynamic network; **LSTM** to predict the topology of the next network respecting local topology and dynamic in the short term
    - use **GAN** to further refine the performance of generative LSTM network
    - **restricted Boltzmann machine** and **graph embedding** -> only applied to unweighted network
    - A. Mozo, B. Ordozgoiti, and S. Gomez, “Forecasting short-term data center network traffic load with convolutional neural networks,” Plos One, vol. 13, no. 2, p. e0191939, 2018.
    - L. Nie, X. Wang, L. Wan, S. Yu, H. Song, and D. Jiang, “Network traffic prediction based on deep belief network and spatiotemporal compressive sensing in wireless mesh backbone networks,” Wireless Communications & Mobile Computing, vol. 2018, pp. 1–10, 2018.


**<u>Influencial Maximization</u>**

* [x] **(very interesting and fresh idea of keeping fair)** (Influence Max/Time)[On the Fairness of Time-Critical Influence Maximization in Social Networks](https://arxiv.org/abs/1905.06618)
    - fairness, try to make the spread of information have same fraction among different social groups iffn social network with budget (the number of seeds) fixed 

* [x] **(very comprehensive for the previous method in influen max)**(Influen max) [TIFIM: A Two-stage Iterative Framework for Influence Maximization in Social Networks](https://www.sciencedirect.com/science/article/abs/pii/S0096300319301602)
    - propose **[TIFIM]**
    - use two stage ierative method to ensure efficiency and accuracy for the influence maximization task

* [x] (Importance to investiage the node change in influen max problem)(Dynamic/Influen max) [Influential Nodes Detection in Dynamic Social Networks](https://link.springer.com/chapter/10.1007/978-3-030-20482-2_6) {local}
    - influential node detection in edge-changing network
        1. Phase 1: Community Detection 
            - > use **Combo**:*Sobolevsky, S., Ratti, C., Campari, R.: General optimization technique for high- quality community detection in complex networks. Phys. Rev. 90, 1–19 (2014)*
        2. Phase 2: Influential Nodes Detection （use diffusion model）
    - <u>**Future**</u>:
        - > predict the change of influential nodes where both nodes and edges evolve in dynamic social networks

* (x) (IM)[Heuristics-based influence maximization for opinion formation in social networks](https://www.sciencedirect.com/science/article/abs/pii/S1568494618300759)

**<u>Other paper</u>**

* [x] **(similarity)** [Exploiting similarities of user friendship networks across social networks for user identification](https://www.sciencedirect.com/science/article/pii/S002002551930756X?via%3Dihub)

**<u>Other than paper:</u>**

* [Matrix Perturbation Theory](http://www.cs.tau.ac.il/~amir1/COURSE2012-2013/perturbationTheory.pdf)

---

### Useful for network analysis research: 

* [Deep Representation Learning for Social Network Analysis](https://www.frontiersin.org/articles/10.3389/fdata.2019.00002/full)
* [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/)
* [[video] Social Network Analysis - From Graph Theory to Applications - Dima Goldenberg - PyCon Israel 2019](https://www.youtube.com/watch?v=px7ff2_Jeqw&list=WL&index=29&t=0s)
* [HOW POWERFUL ARE GRAPH NEURAL NETWORKS?](https://cs.stanford.edu/people/jure/pubs/gin-iclr19.pdf)
* [Social Network Analysis in the field of Learning and Instruction: methodological issues and advances](https://earli.org/sites/default/files/2017-03/ASC2018-web.pdf)
* [Social Network Analysis: ‘How to guide’](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/491572/socnet_howto.pdf)
* [User behavior prediction in social networks using weighted extreme learning machine with distribution optimization](https://www.sciencedirect.com/science/article/pii/S0167739X17307938)
* [Analyzing and inferring human real-life behavior through online social networks with social influence deep learning](https://link.springer.com/article/10.1007/s41109-019-0134-3)

**<u>Mihai Cucuringu (ox)</u>** 

* [ ] [SPONGE: A generalized eigenproblem for clustering signed networks](http://www.stats.ox.ac.uk/~cucuring/signedClustering.pdf)
    - **Clustering**: the average connectivity or similarity between pairs of nodes within the same group is larger than that of pairs of nodes from dif- ferent groups.
* [x] [Anomaly Detection in Networks with Application to Financial Transaction Networks](http://www.stats.ox.ac.uk/~cucuring/anomaly_detection_networks.pdf)
    - Many general network anomaly deteciton reply on  **community detection** -> find the embeddings deviate substantially
    - 


**<u>François Caron (ox)</u>** 

* X. Miscouridou, F. Caron, Y. W. Teh. [Modelling sparsity, heterogeneity, reciprocity and community structure in temporal interaction data](https://papers.nips.cc/paper/7502-modelling-sparsity-heterogeneity-reciprocity-and-community-structure-in-temporal-interaction-data.pdf). Neural Information Processing Systems (NeurIPS'2018), Montreal, Canada, 2018. 

**<u>Rik Sarkar (Edinburgh)</u>:**

* Benedek Rozemberczki, Ryan Davies, Rik Sarkar, Charles Sutton. [GEMSEC: Graph Embedding with Self Clustering](http://homepages.inf.ed.ac.uk/rsarkar/papers/gemsec.pdf), IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM) 2019.
* Panagiota Katsikouli, Maria Sinziana Astefanoaei, Rik Sarkar. [Distributed Mining of Popular Paths in Road Networks](http://homepages.inf.ed.ac.uk/rsarkar/papers/popular_paths_dcoss.pdf), IEEE International Conference on Distributed Computing in Sensor Systems 2018 (DCOSS '18).

**<u>Walid Magdy (Edinburgh)</u>**

* [A Practical Guide for the Effective Evaluation of Twitter User Geolocation](https://arxiv.org/abs/1907.12700)
* [Self-Representation on Twitter Using Emoji Skin Color Modifiers](https://aaai.org/ocs/index.php/ICWSM/ICWSM18/paper/view/17833)

**<u>Kobi Gal (Edinburgh)</u>**

* Avi Segal, Kobi  Gal, Guy Shani and Bracha Shapira. [A difficulty Ranking Approach to Personalization in E-learning](https://arxiv.org/abs/1907.12047). International Journal of Human Computer Studies 130: 261-272, 2019. Supercedes the EDM-14 paper below. 

**<u>Timothy M. Hospedales (Edinburgh)</u>**

* [Feature-Critic Networks for Heterogeneous Domain Generalisation](https://arxiv.org/pdf/1901.11448.pdf)


**<u>Emine Yilmaz (UCL)</u>**

* [User Behaviour and Task Characteristics: A Field Study of Daily Information Behaviour](https://dl.acm.org/citation.cfm?id=3020188)
* [Ranking-based Method for News Stance Detection](http://delivery.acm.org/10.1145/3190000/3186919/p41-zhang.pdf?ip=180.154.9.45&id=3186919&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1571069502_abde5d91c360787561f71b95266fbe84)


**<u>Iván Palomares Carrascosa (Bristol)</u>**

* [Large-scale group decision making model based on social network analysis: Trust relationship-based conflict detection and elimination](https://www.sciencedirect.com/science/article/abs/pii/S0377221718310191?via%3Dihub)
* H. Zhang, I. Palomares, Y.C. Dong, W. Wang. [Managing non-cooperative behaviors in consensus-based multiple attribute group decision making: An approach based on social network analysis](https://doi.org/10.1016/j.knosys.2018.06.008). Knowledge-based Systems, 162, pp. 29-45, 2018.
* Z. Zhang, X. Kou, I. Palomares, W. Yu, J. Gao. [Stable two-sided matching decision making with incomplete fuzzy preference relations: A disappointment theory based approach](https://doi.org/10.1016/j.asoc.2019.105730). Applied Soft Computing. In press: 


### To do for graph & network

* [ ] Morris, C. and Mutzel, P. (2019). Towards a practical k-dimensional weisfeiler-leman algorithm.
arXiv preprint arXiv:1904.01543.

* [ ] Li, Y., Vinyals, O., Dyer, C., Pascanu, R., and Battaglia,
P. Learning deep generative models of graphs. arXiv
preprint arXiv:1803.03324, 2018.

* [ ] Wang, T., Liao, R., Ba, J., and Fidler, S. Nervenet: Learning
structured policy with graph neural networks. In ICLR,
2018a.

* [ ] Al-Rfou, R., Zelle, D., and Perozzi, B. Ddgk: Learning graph representations for deep divergence graph kernels.
arXiv preprint arXiv:1904.09671, 2019.

* [ ] N.Shervashidze,P.Schweitzer,E.J.v.Leeuwen,K.Mehlhorn,andK.M.Borgwardt.Weisfeiler- lehman graph kernels. Journal of Machine Learning Research, 12:2539–2561, 2011. (*Graph Kernel*)

* [ ] Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodola, Jan Svoboda, and Michael M Bronstein. Geometric deep learning on graphs and manifolds using mixture model cnns. arXiv preprint arXiv:1611.08402, 2016. (*MoNet*)


**Knowledge Graph Related**:

* [ ] [Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs](https://www.aclweb.org/anthology/P19-1466)

**Other paper**:

* [ ] T. N. Kipf and M. Welling. Variational graph auto-encoders. In NIPS Workshop on Bayesian
Deep Learning, 2016.

* [ ] Sami Abu-El-Haija. 2017. Proportionate gradient updates with PercentDelta. In arXiv. [graph likelihood]


