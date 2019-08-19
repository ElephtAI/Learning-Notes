## Data Preprocessing and Feature Engineering

* [x] <u>Normalization and Standardization</u> - **Black Gambol - No. 5**
    - [归一化 （Normalization）、标准化 （Standardization）和中心化/零均值化 （Zero-centered）](https://www.jianshu.com/p/95a8f035c86c)
    - [Normalization vs Standardization — Quantitative analysis](https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf)
* [ ] **Dimensionality Reduction Techniques**

    - [x] **Feature Selection** (*Keep the most relevant features*)
        - [<u>Random Forest for Feature Selection</u>](https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f)
        - <u>Missing Value Ratio (缺失值比率)</u>: remove the feature if its missing value ratio is higher than the preset threshold
        - <u>Low Variance Filter (低方差滤波)</u>: remove the features with lowest variance (`need Normalization at first sine variance depends on the range`)
        - <u>High Correlation Filter</u>: remove one of feature pair, if they have high correlation compare to the thresold
        - <u>Backward Feature Elimination</u>: *delete the feature with lowest effect on the model performance at each round*
        - <u>Forward Feature Selection</u>: *choose the feature that can mostly improve the model performance at each round*

    - **Dimensionality Reduction** (*Find new variable with similar information but lower dimension*)
        - **Linear**: 
            - [x] <u>Factor Analysis</u>: [Introduction to Factor Analysis in Python](https://www.datacamp.com/community/tutorials/introduction-factor-analysis)
            - [ ] <u>Principle Component Analysis</u>: Connection to `SVD`
            - [ ] <u>Independent Component Analysis</u>
        - **Non-linear, keep global feature**:
            - [ ] <u>Kernal PCA<u>: Base on **Kernel**
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
    - [ ] <u>Linear Regression</u>
    - [ ] <u>Logistic Regression</u>
    - [ ] Kernel and Regularization (Ridge Regression, Lasso)
    - [ ] Support Vector Machine
    - [ ] Decision Tree
    - [ ] Bagging Algorithm (Random Forests)
    - [ ] Boosting Algorithm
        - [ ] Adaboost
        - [ ] Gradient Boosting Decision Tree
        - [ ] XGBoost
        - [ ] LightGBM
* **Unsupervised Learning**
    - [ ] Unsupervised Learning Summary
    - [ ] Clustering Algorithm Summary


* **Deep Learning Part** (back to the DeepMind Course)
    - [ ] Forward and Backward Propagation 
    - [ ] Activation Function and how to choose
    - [ ] Word2Vec training and GloVe Explaination (Representation Learning)
    - [ ] RNN, LSTM and GRU
        - [ ] Gradient disapper and Gradient explotion
        - [ ] Structure of LSTM and GRU, compare
    - [ ] Encoder and Decoder (seq2seq) - Be able to write from stratch
    - [ ] Attention (Struction and explaination of R-net in Reading comprehension)
    - [ ] Convolusional Neural Network (refer to the DeepMind note)
        - [ ] theory of Convolusion and the equation to calculate
        - [ ] average-pooling, max-pooling, global pooling & reason to pooling
    - Others:
        - [ ] Batch Normalization
        - [ ] How to avoid overfitting

* **Reinforcement Learning Part** (refer to the DeepMind Course and RL book)
* **Graph Learning Part** 
    - AML spectral Learning (Graph Laplacian...)
    - Stanford Analysis of Network course

## Metrics and Ealuation method

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
* Deep Learing Optimizer
    - Adam and many other, how to choose

## Tuning the Hyperparameter

* Tuning Parameter method summary
* Confusion Matrix
* Cross Entropy