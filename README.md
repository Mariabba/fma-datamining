# DataMining2-project

##Module 1 - Introduction, Imbalanced Learning and Anomaly Detection

1-Explore and prepare the dataset. You are allowed to take inspiration from the associated GitHub repository and figure out your personal research perspective (from choosing a subset of variables to the class to predict…). You are welcome in creating new variables and performing all the pre-processing steps the dataset needs.

2-Define one or more (simple) classification tasks and solve it with Decision Tree and KNN. You decide the target variable.

3-Identify the top 1% outliers: adopt at least three different methods from different families (e.g., density-based, angle-based… ) and compare the results. Deal with the outliers by removing them from the dataset or by treating the anomalous variables as missing values and employing replacement techniques. In this second case, you should check that the outliers are not outliers anymore. Justify your choices in every step.

4-Analyze the value distribution of the class to predict with respect to point 2; if it is unbalanced leave it as it is, otherwise turn the dataset into an imbalanced version (e.g., 96% - 4%, for binary classification). Then solve the classification task using the Decision Tree or the KNN by adopting various techniques of imbalanced learning.

5-Draw your conclusions about the techniques adopted in this analysis.

N.B. When “solving the classification task”, remember, (i) to test, when needed, different criteria for the parameter estimation of the algorithms, and (ii) to evaluate the classifiers (e.g., Accuracy, F1, Lift Chart) in order to compare the results obtained with an imbalanced technique against those obtained from using the “original” dataset.

##Module 2 - Advanced Classification Methods

1-Solve the classification task defined in Module 1 (or define new ones) with the other classification methods analyzed during the course: Naive Bayes Classifier, Logistic Regression, Rule-based Classifiers, Support Vector Machines, Neural Networks, Ensemble Methods and evaluate each classifier with the techniques presented in Module 1 (accuracy, precision, recall, F1-score, ROC curve). Perform hyper-parameter tuning phases and justify your choices.

2-Besides the numerical evaluation draw your conclusions about the various classifiers, e.g. for Neural Networks: what are the parameter sets or the convergence criteria which avoid overfitting? For Ensemble classifiers how the number of base models impacts the classification performance? For any classifier which is the minimum amount of data required to guarantee an acceptable level of performance? Is this level the same for any classifier? What is revealing the feature importance of Random Forests?
N.B. When “solving the classification task”, remember, (i) to test, when needed, different criteria for the parameter estimation of the algorithms, and (ii) to evaluate the classifiers (e.g., Accuracy, F1, Lift Chart) in order to compare the results obtained with an imbalanced technique against those obtained from using the “original” dataset.
