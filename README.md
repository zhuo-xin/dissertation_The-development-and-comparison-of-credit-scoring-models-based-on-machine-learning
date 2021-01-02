The development and comparison of credit scoring models based on machine learning methods

Credit scoring as a statistical method to evaluate the probability that loan applicants would default is widely deployed in financial market, such as mortgage and insurance. The significant part of credit scoring is building models, in practice, logistic regression, decision tree and neural network models are relatively popular, as computer power increasing, new methods are created to build efficient models, especially in machine learning field. 

This study aims to review some traditional methods and try to build new credit scoring models to improve the accuracy based on Kaggle simulated data, there are two kinds of methods used: single classification methods and ensemble learning methods. Specifically, single classification methods include Decision Tree (DT), Support Vector Machine (SVM), K Nearest neighbour (KNN), Naïve Bayes (NB), Logistic Regression (LR) and Multi-layer Perceptron (MLP). Ensemble learning methods include Gradient Boosting Decision Tree (GBDT), Random Forest (RF), Adaboost, and Stacking models.

The result indicates that the performance of most ensemble learning methods is slightly better than other methods used in this research, especially for stacking model 4 (ST4), which is combined with GBDT, RF, Adaboost (based on DT), DT, MLP and LR. ST4 was in a leading position in terms of all performance evaluation indicators such as accuracy, precision, AUC value and so on, which may be a new efficient model to be applied in credit scoring theoretically. Further research regarding applying more machine learning methods would be interesting, and it is significant to build models on larger real data set.

Related documents include:
1. data preprocessing.py is used for data preprocessing, missing value filling and outlier detection processing
2. single method.py performs a single classification method on the data set, including decision tree (DT), support vector machine (SVM), K nearest neighbor (KNN), naive Bayes (NB), logistic regression (LR) And Multilayer Perceptron (MLP)
3. ensemble learning methods.py is used to perform integrated learning methods on data, including gradient boosting decision tree (GBDT), random forest (RF), Adaboost and stacking models
4. summary of results.py summarizes and visualizes all model results, and compares and shows which algorithms perform better
5. credit_risk_dataset.csv is the data set used in this research, from Kaggle, address: https://www.kaggle.com/laotse/credit-risk-dataset

基于信用风险评分的机器学习模型比较和新模型开发

信用评分是一种统计方法，用于评估贷款申请人违约的可能性，已广泛应用于抵押贷款和保险等金融市场。 
信用评分的重要部分是建立模型，在实践中，逻辑回归，决策树和神经网络模型相对流行，随着计算机计算能力的增强，研究者们创建了新的方法来建立有效的模型，尤其是在机器学习领域。

本研究旨在回顾一些传统方法，并尝试基于Kaggle模拟数据建立新的信用评分模型以提高准确性，其中使用了两种方法：单一分类方法和集成学习方法。
具体来说，单一分类方法包括决策树（DT），支持向量机（SVM），K最近邻（KNN），朴素贝叶斯（NB），逻辑回归（LR）和多层感知器（MLP）。 
集成学习方法包括梯度提升决策树（GBDT），随机森林（RF），Adaboost和stacking模型。

结果表明，大多数集成学习方法的性能略优于本研究中使用的其他方法，特别是对于堆叠模型4（ST4）组成部分包括：GBDT，RF，Adaboost（基于DT），DT，MLP和 LR。
ST4在所有性能评估指标（如准确性，精度，AUC值等）方面处于领先地位，这可能是一种理论上可用于信用评分的新型高效模型。 
关于应用更多机器学习方法的进一步研究将很有趣，并且在更大的真实数据集上构建模型具有重要意义。

相关文件包括：
1. data preprocessing.py 用于数据预处理，缺失值填补和异常值探测处理
2. single method.py 对数据集进行单一分类方法处理，包括了决策树（DT），支持向量机（SVM），K最近邻（KNN），朴素贝叶斯（NB），逻辑回归（LR）和多层感知器（MLP）
3. ensemble learning methods.py 用于对数据进行集成学习方法，包括了梯度提升决策树（GBDT），随机森林（RF），Adaboost和stacking模型
4. summary of results.py 对所有模型结果进行汇总总结和可视化处理，对比展示哪些算法效果更好
5. credit_risk_dataset.csv 是针对本研究所使用的数据集，来自于Kaggle，地址：https://www.kaggle.com/laotse/credit-risk-dataset
