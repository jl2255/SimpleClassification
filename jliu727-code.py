
# coding: utf-8

# In[4]:

# Dataset 1
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import neighbors

# import mushroom dataset
mushrooms = pd.read_csv('mushrooms.csv')
mr_new = pd.get_dummies(mushrooms.loc[:, mushrooms.columns != 'class'])
mr_new['class'] = mushrooms.loc[:, mushrooms.columns == 'class']
train, test = train_test_split(mr_new, test_size = 0.5, random_state = 100)
# train set
train_x = train.loc[:, train.columns != 'class']
train_y = train.loc[:, train.columns == 'class']
test_x = test.loc[:, test.columns != 'class']
test_y = test.loc[:, test.columns == 'class']

# Decision Tree
tree_model = tree.DecisionTreeClassifier(max_depth = 8 ,criterion = 'entropy')
tree_model.fit(train_x, train_y)
tree_model.score(train_x, train_y)
tree_model.score(test_x, test_y)
# randomly add some noises to the dataset
rand_mr = mr_new.copy()
p=rand_mr.sample(math.ceil(len(rand_mr)*0.05))
p['class'] = 'p'
rand_mr.update(p)
e=rand_mr.sample(math.ceil(len(rand_mr)*0.05))
e['class'] = 'e'
rand_mr.update(e)
#set train size
rand_train, rand_test = train_test_split(rand_mr, test_size = 0.1, random_state = 100)
rd_train_x = rand_train.loc[:, rand_train.columns != 'class']
rd_train_y = rand_train.loc[:, rand_train.columns == 'class']
rd_test_x = rand_test.loc[:, rand_test.columns != 'class']
rd_test_y = rand_test.loc[:, rand_test.columns == 'class']
tree_model2 = tree.DecisionTreeClassifier(max_depth = 5, criterion = 'entropy')
tree_model2.fit(rd_train_x, rd_train_y)
tree_model2.score(rd_train_x, rd_train_y)
tree_model2.score(rd_test_x, rd_test_y)

# Neural Network
rand_train, rand_test = train_test_split(rand_mr, test_size = 0.1, random_state = 100)
rd_train_x = rand_train.loc[:, rand_train.columns != 'class']
rd_train_y = rand_train.loc[:, rand_train.columns == 'class']
rd_test_x = rand_test.loc[:, rand_test.columns != 'class']
rd_test_y = rand_test.loc[:, rand_test.columns == 'class']
nn_model = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes=(8), alpha = 0.001, random_state = 1)
nn_model.fit(rd_train_x, rd_train_y)
nn_model.score(rd_train_x, rd_train_y)
nn_model.score(rd_test_x, rd_test_y)

# Boosting
rand_train, rand_test = train_test_split(rand_mr, test_size = 0.1, random_state = 100)
rd_train_x = rand_train.loc[:, rand_train.columns != 'class']
rd_train_y = rand_train.loc[:, rand_train.columns == 'class']
rd_test_x = rand_test.loc[:, rand_test.columns != 'class']
rd_test_y = rand_test.loc[:, rand_test.columns == 'class']
boosting_model = AdaBoostClassifier(n_estimators=100, base_estimator = tree.DecisionTreeClassifier(max_depth=1))
boosting_model.fit(rd_train_x, rd_train_y)
boosting_model.score(rd_train_x, rd_train_y)
boosting_model.score(rd_test_x, rd_test_y)

#Support Vector Machine
rand_train, rand_test = train_test_split(rand_mr, test_size = 0.1, random_state = 100)
rd_train_x = rand_train.loc[:, rand_train.columns != 'class']
rd_train_y = rand_train.loc[:, rand_train.columns == 'class']
rd_test_x = rand_test.loc[:, rand_test.columns != 'class']
rd_test_y = rand_test.loc[:, rand_test.columns == 'class']

SVM_model_linear = svm.SVC(kernel = 'linear')
SVM_model_sigmoid = svm.SVC(kernel = 'sigmoid')

SVM_model_linear.fit(rd_train_x, rd_train_y)
SVM_model_sigmoid.fit(rd_train_x, rd_train_y)

SVM_model_linear.score(rd_train_x, rd_train_y)
SVM_model_linear.score(rd_test_x, rd_test_y)
SVM_model_sigmoid.score(rd_train_x, rd_train_y)
SVM_model_sigmoid.score(rd_test_x, rd_test_y)

# kNN
kNN_model_5 = neighbors.KNeighborsClassifier(5, weights = 'uniform')
kNN_model_20 = neighbors.KNeighborsClassifier(20, weights = 'uniform')
rand_train, rand_test = train_test_split(rand_mr, test_size = 0.1, random_state = 100)
rd_train_x = rand_train.loc[:, rand_train.columns != 'class']
rd_train_y = rand_train.loc[:, rand_train.columns == 'class']
rd_test_x = rand_test.loc[:, rand_test.columns != 'class']
rd_test_y = rand_test.loc[:, rand_test.columns == 'class']
kNN_model_5.fit(rd_train_x, rd_train_y)
kNN_model_20.fit(rd_train_x, rd_train_y)
kNN_model_5.score(rd_train_x, rd_train_y)
kNN_model_5.score(rd_test_x, rd_test_y)
kNN_model_20.score(rd_train_x, rd_train_y)
kNN_model_20.score(rd_test_x, rd_test_y)


# In[3]:

# Dataset 2

# import collge ranking dataset
cwur = pd.read_csv('cwurData_new.csv')
country = pd.get_dummies(cwur.loc[:, cwur.columns == 'country'])
cwur_new = cwur.loc[:, cwur.columns != 'country']
cwur_new = pd.concat([cwur_new, country], axis = 1)
cwur_new.loc[cwur_new['score'] <= 45, 'score'] = 0
cwur_new.loc[cwur_new['score'] > 45, 'score'] = 1

#Decision Tree
train, test = train_test_split(cwur_new, test_size = 0.1, random_state = 100)
train_x = train.loc[:, train.columns != 'score']
train_y = train.loc[:, train.columns == 'score']
test_x = test.loc[:, test.columns != 'score']
test_y = test.loc[:, test.columns == 'score']
tree_model = tree.DecisionTreeClassifier(max_depth = 15)
tree_model.fit(train_x, train_y)
tree_model.score(train_x, train_y)
tree_model.score(test_x, test_y)

# Neural Networks
train, test = train_test_split(cwur_new, test_size = 0.1, random_state = 100)
train_x = train.loc[:, train.columns != 'score']
train_y = train.loc[:, train.columns == 'score']
test_x = test.loc[:, test.columns != 'score']
test_y = test.loc[:, test.columns == 'score']
nn_model = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes=(3), alpha = 0.001, random_state = 1)
nn_model.fit(train_x, train_y)
nn_model.score(train_x, train_y)
nn_model.score(test_x, test_y)

# Boosting
train, test = train_test_split(cwur_new, test_size = 0.1, random_state = 100)
train_x = train.loc[:, train.columns != 'score']
train_y = train.loc[:, train.columns == 'score']
test_x = test.loc[:, test.columns != 'score']
test_y = test.loc[:, test.columns == 'score']
boosting_model = AdaBoostClassifier(n_estimators=100, base_estimator = tree.DecisionTreeClassifier(max_depth=1))
boosting_model.fit(train_x, train_y)
boosting_model.score(train_x, train_y)
boosting_model.score(test_x, test_y)

#Support Vector Machine
train, test = train_test_split(cwur_new, test_size = 0.8, random_state = 100)
train_x = train.loc[:, train.columns != 'score']
train_y = train.loc[:, train.columns == 'score']
test_x = test.loc[:, test.columns != 'score']
test_y = test.loc[:, test.columns == 'score']
SVM_model_linear = svm.SVC(kernel = 'linear')
SVM_model_sigmoid = svm.SVC(kernel = 'sigmoid')
SVM_model_linear.fit(train_x, train_y)
SVM_model_sigmoid.fit(train_x, train_y)
SVM_model_linear.score(train_x, train_y)
SVM_model_sigmoid.score(train_x, train_y)
SVM_model_linear.score(test_x, test_y)
SVM_model_sigmoid.score(test_x, test_y)

#kNN
kNN_model_5 = neighbors.KNeighborsClassifier(5, weights = 'uniform')
kNN_model_20 = neighbors.KNeighborsClassifier(20, weights = 'uniform')
train, test = train_test_split(cwur_new, test_size = 0.2, random_state = 100)
train_x = train.loc[:, train.columns != 'score']
train_y = train.loc[:, train.columns == 'score']
test_x = test.loc[:, test.columns != 'score']
test_y = test.loc[:, test.columns == 'score']
kNN_model_5.fit(train_x, train_y)
kNN_model_20.fit(train_x, train_y)
kNN_model_5.score(train_x, train_y)
kNN_model_5.score(test_x, test_y)
kNN_model_20.score(train_x, train_y)
kNN_model_20.score(test_x, test_y)

