## General to read reference:

* [神经网络的激活函数总结](https://zhuanlan.zhihu.com/p/40903328): **links to useful pages at the bottom**
* [推荐系统从零单排系列(六)—Word2Vec优化策略层次Softmax与负采样](https://zhuanlan.zhihu.com/p/66417229): **links to the series**
* [Graph Neural Network（GNN）最全资源整理分享 - 知乎](https://zhuanlan.zhihu.com/p/73052234)

## Data Preprocessing and Feature Engineering

* [x] <u>Normalization and Standardization</u> - `<BGI - No.5>`
    - [归一化 （Normalization）、标准化 （Standardization）和中心化/零均值化 （Zero-centered）](https://www.jianshu.com/p/95a8f035c86c)
    - [Normalization vs Standardization — Quantitative analysis](https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf)
* [ ] **Dimensionality Reduction Techniques**

    - [x] **Feature Selection** (*Keep the most relevant features*)
        - [<u>Random Forest for Feature Selection</u>](https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f): `<BGI - No.6.1>`
        - <u>Missing Value Ratio (缺失值比率)</u>: remove the feature if its missing value ratio is higher than the preset threshold
        - <u>Low Variance Filter (低方差滤波)</u>: remove the features with lowest variance (`need Normalization at first sine variance depends on the range`)
        - <u>High Correlation Filter</u>: remove one of feature pair, if they have high correlation compare to the thresold
        - <u>Backward Feature Elimination</u>: *delete the feature with lowest effect on the model performance at each round*
        - <u>Forward Feature Selection</u>: *choose the feature that can mostly improve the model performance at each round*

    - **Dimensionality Reduction** (*Find new variable with similar information but lower dimension*)
        - **Linear**: 
            - [x] <u>Factor Analysis</u>: [Introduction to Factor Analysis in Python](https://www.datacamp.com/community/tutorials/introduction-factor-analysis): `<BGI - No.6.2>`
            - [x] <u>Principle Component Analysis</u>: connection to **SVD** `<BGI - No.6.3>`
                - [PCA和SVD的关系 - CSDN](https://blog.csdn.net/billbliss/article/details/80451748)
            - [x] <u>Independent Component Analysis</u>: [独立成分分析 ( ICA ) 与主成分分析 ( PCA ) 的区别在哪里？](https://www.zhihu.com/question/28845451) `<BGI - No.6.5>`
        - **Non-linear, keep global feature**:
            - [x] <u>Kernal PCA</u>: `<BGI - No.6.6>`
                - [降维算法总结比较（三）- Kenel PCA](https://zhuanlan.zhihu.com/p/25097144) 
                - [Kernel tricks and nonlinear dimensionality reduction via RBF kernel PCA](https://sebastianraschka.com/Articles/2014_kernel_pca.html#gaussian-radial-basis-function-rbf-kernel-pca)
            - [x] <u>Auto-Encoder</u>: Base on **Neural Network** `<BGI - No.17.1-3>`
                - [什么是自编码 Autoencoder - 莫烦](https://zhuanlan.zhihu.com/p/24813602)
                - [当我们在谈论 Deep Learning：AutoEncoder 及其相关模型 - 知乎](https://zhuanlan.zhihu.com/p/27865705)
            - <u>Multidimensionaly Scaling (MDS)</u>
            - <u>Isomap</u>
            - <u>Diffusion Map</u>

        - **Non-linear, keep local feature**:
            - <u>Locally Linear Embedding (LLE, 局部线性嵌入)</u>
            - <u>Laplacian Eigenmaps</u>
        
        - **Others**:
            - <u>T-SNE</u>
            - <u>UMAP</u>

    - [ ] Extension of SVD (-> PCA -> LSI -> LDA)
        - **what is Eigenvalue**: <u>*Eigenvectors are used for understanding linear transformations. In data analysis, we usually calculate the eigenvectors for a correlation or covariance matrix. Eigenvectors are the directions along which a particular linear transformation acts by flipping, compressing or stretching. Eigenvalue can be referred to as the strength of the transformation in the direction of eigenvector or the factor by which the compression occurs.*</u>
        - [Latent Semantic Analysis (LSA) Tutorial](https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/)

    - Referenccs: 
        - [降维方法总结 - CSDN](https://blog.csdn.net/qq_28266311/article/details/93342737)
        - [10种常用降维算法源代码(python) - 知乎](https://zhuanlan.zhihu.com/p/68754729)
        - [降维算法总结比较（二）- Laplacian Eigenmaps - 知乎](https://zhuanlan.zhihu.com/p/25096844)
        - [降维算法总结比较（三）- Kenel PCA](https://zhuanlan.zhihu.com/p/25097144)
        - [奇异值的物理意义是什么？强大的矩阵奇异值分解(SVD)及其应用](https://blog.csdn.net/c2a2o2/article/details/70159320)
* Tokenization
* Processing the imbalancing data


## Learning Algorithm

* **Supervised Learning Part**
    - [x] <u>Linear Regression:</u> `<BGI - No.7.1>`
    - [x] <u>Logistic Regression:</u> `<BGI - No.7.2>`
    - [x] <u>Kernel and Regularization</u> (Ridge Regression, Lasso): `<BGI - No.8.1-3>` 
        - L1 compared to L2 regularization: [L1,L2,L0区别，为什么可以防止过拟合 - 简书](https://www.jianshu.com/p/475d2c3197d2)
        - Ridge vs Lasso: [变量的选择——Lasso&Ridge&ElasticNet](https://www.cnblogs.com/mengnan/p/9307615.html)
    - Support Vector Machine
    - [x] <u>Decision Tree</u> `<BGI - No.15.1-4>` 
    - [x] <u>Bagging Algorithm (Random Forests)</u> `<BGI - No.16.1, No.16.3>`
    - [ ] Boosting Algorithm `<BGI - No.16.2>`
        - [x] <u>Adaboost</u> [机器学习算法中GBDT与Adaboost的区别与联系是什么？](https://www.zhihu.com/question/54626685/answer/140610056) `<BGI - No.16.3>`
        - [x] <u>Gradient Boosting Decision Tree</u> `<BGI - No.16.4>`
            - [当我们在谈论GBDT：从 AdaBoost 到 Gradient Boosting](https://zhuanlan.zhihu.com/p/25096501?refer=data-miner)
            - [当我们在谈论GBDT：Gradient Boosting 用于分类与回归](https://zhuanlan.zhihu.com/p/25257856)
            - [机器学习算法中GBDT与Adaboost的区别与联系是什么？](https://www.zhihu.com/question/54626685/answer/140610056)
        - [x] <u>XGBoost</u> `<BGI - No.16.5>`
            - [决策树、随机森林、bagging、boosting、Adaboost、GBDT、XGBoost总结 - 知乎](https://zhuanlan.zhihu.com/p/75468124)
            - [机器学习算法中 GBDT 和 XGBOOST 的区别有哪些？ - 知乎](https://www.zhihu.com/question/41354392/answer/124274741)
        - [ ] LightGBM
        - [ ] CatBoost

        - **References**: 
            - [决策树、随机森林、bagging、boosting、Adaboost、GBDT、XGBoost总结 - 知乎](https://zhuanlan.zhihu.com/p/75468124)

    - [ ] Gaussian Process


* **Unsupervised Learning**
    - [ ] Unsupervised Learning Summary
    - [ ] Clustering Algorithm Summary
    - [ ] Topic Modelling: NMF (AML notes), Latent Dirichlet Allocation (LDA)
    - [ ] Latent Semantic AnalysisLatent Semantic Indexing (LSI)


* **Deep Learning Part** (back to the DeepMind Course)
    - [x] <u>Forward and Backward Propagation</u>: [反向传播算法推导-全连接神经网络 - SIGAI](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484439&idx=1&sn=4fa8c71ae9cb777d6e97ebd0dd8672e7&chksm=fdb69980cac110960e08c63061e0719a8dc7945606eeef460404dc2eb21b4f5bdb434fb56f92&scene=21#wechat_redirect)
    - [x] <u>Activation Function and how to choose</u>: `<BGI - No.9.1-2>`
        - keep the nonlinearity of the neural network
        - avoid **gradient vanishing** and **gradient explosion**
        - [理解神经网络的激活函数](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247483977&idx=1&sn=401b211bf72bc70f733d6ac90f7352cc&chksm=fdb69fdecac116c81aad9e5adae42142d67f50258106f501af07dc651d2c1473c52fad8678c3&scene=21#wechat_redirect)
        - Saturation activation function: [Noisy Activation Functions](https://arxiv.org/pdf/1603.00391.pdf)
        
    - [ ] Word level representation Learning (Representation Learning):
        - [x] <u>Bag of words model</u>: `<BGI - No.10.1>`
            - **Problem**: `if large vocab -> dimension too large`; `ignore the word order`; `cannot represent semantic level meaning / in a context`
        - [x] <u>Word2Vec (**local context**)</u> (*CBoW & Skip-Gram, Hierachical Softmax & Negative Sampling*): `<BGI - No.10.2>`
            - [ ] **Problem**: `no overal cooccur (only local context)`, `word have only one vector`
            - [ ] form a huffman tree in python: [数据结构和算法——Huffman树和Huffman编码 - CSDN](https://blog.csdn.net/google19890102/article/details/54848262)
            - [x] how is negative sampling work in [the word2vec paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
            - [一篇浅显易懂的word2vec原理讲解 - 知乎](https://zhuanlan.zhihu.com/p/44599645)
            - [WORD2VEC原理及使用方法总结 - 知乎](https://zhuanlan.zhihu.com/p/31319965)
            - [数据结构和算法——Huffman树和Huffman编码 - CSDN](https://blog.csdn.net/google19890102/article/details/54848262)
        - [x] <u>GloVe</u>: [通俗易懂理解——Glove算法原理](https://zhuanlan.zhihu.com/p/42073620): `<BGI - No.10.3>`
        - [x] <u>Tf-idf</u>: `<BGI - No.10.4>`
            - **Problem**: `cannot only use frequency`, `ignore word order`, `no importance in context`
        - [x] Latent Semantic Analysis (LSA): [Latent Semantic Analysis (LSA) Tutorial](https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/): `<BGI - No.10.5>`

        - [x] **Doc2Vec**: `<BGI - No.13.1>`
            - [基于Doc2vec训练句子向量](https://zhuanlan.zhihu.com/p/36886191)

        - [ ] other representation learning for nlp: ELMO, GPT, Transformer, Bert
            - [ ] **Representation** learning of the flower book
        
    - [x] RNN, LSTM and GRU `<BGI - No.11.1-3>`
        - [x] Gradient vanishing and Gradient explosion
        - [x] Structure of LSTM and GRU, compare
        
    - [x] Convolusional Neural Network (refer to the DeepMind note) `<BGI - No.12.1-2>`
        - [x] theory of Convolusion and the equation to calculate
        - [x] average-pooling, max-pooling, global pooling & reason to pooling
        - AlexNet
        - VGGNet
        - ResNet
        - Inception Net 

    - [x] Encoder and Decoder (seq2seq) - Be able to write from stratch `<BGI - No.14.1>`
        - [从Encoder到Decoder实现Seq2Seq模型 - 知乎](https://zhuanlan.zhihu.com/p/27608348); [Github](https://github.com/NELSONZHAO/zhihu/blob/master/basic_seq2seq/Seq2seq_char.ipynb)

    - [x] Attention (Struction and explaination of R-net in Reading comprehension) `<BGI - No.14.2>`      
        - [基于Keras框架实现加入Attention与BiRNN的机器翻译模型 - 知乎](https://zhuanlan.zhihu.com/p/37290775)
        - [经典算法·从seq2seq、attention到transformer - 知乎](https://zhuanlan.zhihu.com/p/54368798)
        - [完全图解RNN、RNN变体、Seq2Seq、Attention机制 - 知乎](https://zhuanlan.zhihu.com/p/28054589)

    - [ ] Representation Learning - <u>The chapter in the FLower book</u>

    - [ ] R-net structure and explaination
    - [ ] NL2SQL structure and explaination

    - Generative Model
        - [ ] Probabilistic PCA 
        - [x] Variational Auto-Encoder (VAE)
            - [当我们在谈论 Deep Learning：AutoEncoder 及其相关模型](https://zhuanlan.zhihu.com/p/27865705)
            - [变分自编码器（VAEs）](https://zhuanlan.zhihu.com/p/25401928)
            - [Tutorial - What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
            - [Tensorflow - code example](https://github.com/y0ast/VAE-TensorFlow); [Torch - code example](https://github.com/y0ast/VAE-Torch)
            - [Auto-Encoding Variational Bayes - original paper](https://arxiv.org/abs/1312.6114)

        - [ ] GAN
        - References:
            - [当我们在谈论 Deep Learning：AutoEncoder 及其相关模型 - 知乎](https://zhuanlan.zhihu.com/p/27865705)
            - [漫谈概率 PCA 和变分自编码器 - 机器之心](jiqizhixin.com/articles/2018-08-16-7)


    - Others:
        - [x] Batch Normalization `<BGI - No.12.3>`
        - [ ] How to avoid overfitting


* **Transfer Learning**

* **Reinforcement Learning Part** (refer to the DeepMind Course and RL book)



*

* **Graph Learning Part** 
    - AML spectral Learning (Graph Laplacian...)
    - Stanford Analysis of Network course

    - [ ] **Knowledge Graph**
        - [ACL 2019 知识图谱的全方位总结](https://www.leiphone.com/news/201908/13I5DsoFpwnLC6sA.html)

## Metrics and Ealuation method


* Mean Squared Error (MSE) `<BGI - No.18.1>`
* Mean Absolute Error (MAE) `<BGI - No.18.2>`
* Huber Loss `<BGI - No.18.3>`
* Cross entropy (two classification and multi classification) `<BGI - No.18.5>`
* Hinge Loss `<BGI - No.18.6>`
* **KL - divergence** `<BGI - No.18.7>`
    - [Making sense of the Kullback–Leibler (KL) Divergence](https://medium.com/@cotra.marko/making-sense-of-the-kullback-leibler-kl-divergence-b0d57ee10e0a)
    - KL divergence between two normal distribution [多变量高斯分布之间的KL散度（KL Divergence）- CSDN](https://blog.csdn.net/wangpeng138375/article/details/78060753)
* Classification Accuracy `<BGI - No.18.8>`
* Confusion Map (True Positive, True Negative, False Positive, False Negative) `<BGI - No.18.9>`
* Area Under Curve `<BGI - No.18.10>`
    - True Positive Rate (sensitivity)
    - False Positive Rate (Specificity)
* Precision, Recall, F1 `<BGI - No.18.11>`
* Error Rate `<BGI - No.18.12>`

* Reference: 
    - [机器学习常用损失函数小结 - 知乎](https://zhuanlan.zhihu.com/p/77686118)
    - [Metrics to Evaluate your Machine Learning Algorithm](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234)

## Optimization part

* Optimization Theory
    - Gradient Descent (stochastic, mini-batch or others)
    - Revise the Optimization part of the AML notes
    - [理解凸优化](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484439&idx=1&sn=4fa8c71ae9cb777d6e97ebd0dd8672e7&chksm=fdb69980cac110960e08c63061e0719a8dc7945606eeef460404dc2eb21b4f5bdb434fb56f92&scene=21#wechat_redirect)
* Deep Learing Optimizer
    - Adam and many other, how to choose

## Tuning the Hyperparameter

* Tuning Parameter method summary
* Confusion Matrix
* Cross Entropy