# 项目简介
    本项目采取通过使用 Python 编程环境实现决策树、逻辑回归、随机森林、梯度提升树等多种模型来对 5G 用户进行预测。
    任务的核心是基于用户的基本信息和通信相关数据，如用户话费信息、流量、活跃行为、套餐类型、区域信息等特征字段，通
    过训练数据集训练模型，预测测试集中每个样本是否为5G 用户。本次任务评估的标准是 AUC（Area Under the Curve
    ），即分数越高，效果越好。

# 环境依赖
### python 版本
    Python 3.8
### python 依赖包:
    (1) pandas
    (2) sklearn
    (3) matplotlib
    (4) numpy
    (5) joblib
### IDE:
    PyCharm 2023.2.5

# 目录结构描述

```
    │  5G 用户预测分析报告.docx                             // 实验报告
    │  DecisionTree.ipynb                                // 决策树模型的ipynb文件  
    │  DecisionTree.py                                   // 决策树模型的文件  
    │  LogisticRegression.py                             // 线性回归模型的ipynb文件  
    │  logistic_regression.ipynb                         // 线性回归模型的文件  
    │  RandomForest.py                                   // 随机森林模型的ipynb文件   
    │  random_forest.jpynb.ipynb                         // 随机森林模型的文件  
    │  ReadMe.md                                         // 帮助文档  
    │  xgboost.ipynb                                     // 梯度提升树模型的ipynb文件  
    │  xgboost.py                                        // 梯度提升树模型的文件  
    │  
    ├─.idea                                              //Pycharm 配置文件  
    │          
    ├─.venv                                              //python依赖包  
    │          
    ├─data
    │      train.csv                                     // 原始数据  
    │      
    └─results                                            // 存放各个模型运行结果
        ├─decision_tree_results                          // 决策树运行结果
        │      confusion_matrix_tree.png                 // 混淆矩阵图片
        │      decision_tree_model.pkl                   // 模型文件
        │      label_encoders.pkl                        // 标签编码器
        │      roc_curve_tree.png                        // ROC曲线图
        │      scaler.pkl                                // 定标器模型
        │      test_predictions_tree.csv                 // 预测结果
        │      
        ├─gradient_boosting_results
        │      confusion_matrix_gb.png                   // 混淆矩阵图片
        │      gb_model.pkl                              // 模型文件
        │      roc_curve_gb.png                          // ROC曲线图
        │      test_predictions_gb.csv                   // 预测结果
        │      
        ├─logistic_regression_results
        │      confusion_matrix_lr.png                   // 混淆矩阵图片
        │      lr_model.pkl                              // 模型文件
        │      roc_curve_lr.png                          // ROC曲线图
        │      test_predictions_lr.csv                   // 预测结果
        │      
        └─random_forest_results
                confusion_matrix_rf.png                  // 混淆矩阵图片
                rf_model.pkl                             // 模型文件
                roc_curve_rf.png                         // ROC曲线图
                test_predictions_rf.csv                  // 预测结果
```
# 作者列表
    
# 版本更新内容摘要
###### v 1.0.0
    实现决策树、线性回归、随机森林、梯度递降树四个模型的实现
###### v 1.1.0
    添加了notebook文件
###### v 1.2.0
    添加了对应的英文注解