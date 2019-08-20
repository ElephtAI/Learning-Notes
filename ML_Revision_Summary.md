## General to read reference:

* [神经网络的激活函数总结](https://zhuanlan.zhihu.com/p/40903328): **links to useful pages at the bottom**
* [推荐系统从零单排系列(六)—Word2Vec优化策略层次Softmax与负采样](https://zhuanlan.zhihu.com/p/66417229): **links to the series**

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
            - [ ] <u>Auto-Encoder</u>: Base on **Neural Network**
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
    - [ ] Decision Tree
    - [ ] Bagging Algorithm (Random Forests)
    - [ ] Boosting Algorithm
        - [ ] Adaboost
        - [ ] Gradient Boosting Decision Tree
        - [ ] XGBoost
        - [ ] LightGBM
    - [ ] Gaussian Process
* **Unsupervised Learning**
    - [ ] Unsupervised Learning Summary
    - [ ] Clustering Algorithm Summary
    - [ ] Topic Modelling: NMF (AML notes)
    - [ ] Latent Semantic Indexing (LSI)
    - [ ] Latent Dirichlet Allocation (LDA)


* **Deep Learning Part** (back to the DeepMind Course)
    - [x] <u>Forward and Backward Propagation</u>: [反向传播算法推导-全连接神经网络 - SIGAI](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484439&idx=1&sn=4fa8c71ae9cb777d6e97ebd0dd8672e7&chksm=fdb69980cac110960e08c63061e0719a8dc7945606eeef460404dc2eb21b4f5bdb434fb56f92&scene=21#wechat_redirect)
    - [x] <u>Activation Function and how to choose</u>: `<BGI - No.9.1-2>`
        - keep the nonlinearity of the neural network
        - avoid **gradient vanishing** and **gradient explosion**
        - [理解神经网络的激活函数](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247483977&idx=1&sn=401b211bf72bc70f733d6ac90f7352cc&chksm=fdb69fdecac116c81aad9e5adae42142d67f50258106f501af07dc651d2c1473c52fad8678c3&scene=21#wechat_redirect)
        - Saturation activation function: [Noisy Activation Functions](https://arxiv.org/pdf/1603.00391.pdf)
    - [ ] Word level representation Learning (Representation Learning):
        - [x] <u>Bag of words model</u>: `<BGI - No.10.1>`
        - [ ] <u>Word2Vec</u> (*CBoW & Skip-Gram, Hierachical Softmax & Negative Sampling*): `<BGI - No.10.2>`
            - [一篇浅显易懂的word2vec原理讲解 - 知乎](https://zhuanlan.zhihu.com/p/44599645)
            - [WORD2VEC原理及使用方法总结 - 知乎](https://zhuanlan.zhihu.com/p/31319965)
        - [ ] GloVe
        - [ ] **Doc2Vec**
        - [ ] Tf-idf: **advantages and disadvantages**
        - [ ] other representation learning for nlp: GPT, Transformer, Bert
    - [ ] RNN, LSTM and GRU
        - [ ] Gradient disapper and Gradient explotion
        - [ ] Structure of LSTM and GRU, compare
    - [ ] Convolusional Neural Network (refer to the DeepMind note)
        - [ ] theory of Convolusion and the equation to calculate
        - [ ] average-pooling, max-pooling, global pooling & reason to pooling
    - [ ] Encoder and Decoder (seq2seq) - Be able to write from stratch
    - [ ] Attention (Struction and explaination of R-net in Reading comprehension)

    - Others:
        - [ ] Batch Normalization
        - [ ] How to avoid overfitting

* **Reinforcement Learning Part** (refer to the DeepMind Course and RL book)
* **Graph Learning Part** 
    - AML spectral Learning (Graph Laplacian...)
    - Stanford Analysis of Network course

## Metrics and Ealuation method

* [ ] **when to choose each loss function**

* MLE, MAP
* Mean square error, Mean Abosolute Error
* Classification Accuracy
* Precision, Recall, F1 and AUC
* Logrithmic Loss

* Reference: 
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