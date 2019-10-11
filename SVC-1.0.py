from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')

# LOAD DATA
data_raw = pd.read_csv(
    '/Users/deyxie/Data/BankdataCompetition/Train_dataset.csv')

# CHECK THE DATA SHAPE AND BASIC INFORMATION
print(data_raw.shape)
print(data_raw.describe())

# check the data balance
the_1class_num = sum(data_raw['target'])  # 12059
the_0class_num = 120000 - the_1class_num
print("class 1: ", the_1class_num)
print("class 0: ", the_0class_num)


# Sampling for data balance
# using tran_id  to find the right rows
data_12059_1 = data_raw.loc[data_raw['target'] ==
                            1, :].sample(n=the_1class_num, random_state=123)
data_12059_0 = data_raw.loc[data_raw['target'] ==
                            0, :].sample(n=the_1class_num, random_state=123)
data_balanced = pd.concat([data_12059_1, data_12059_0], axis=0)


# Split the data into train and test
data_train = data_balanced.iloc[:, 2:202]
data_target = data_balanced['target']
# check the shape
print(data_train.shape)
print(data_target.shape)


# Normalization
myStand = StandardScaler().fit(data_train)
data_train_stand = myStand.transform(data_train)


# PCA
myPCA = PCA(n_components=30).fit(data_train_stand)
data_train_stand_pca = myPCA.transform(data_train_stand)
data_train_stand_pca.shape


# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    data_train_stand_pca, data_target, test_size=0.2, random_state=2)


# The model
print("Starting training")
model_svc = SVC(
    C=1.0,
    kernel='rbf',
    gamma='auto',
    random_state=123
).fit(x_train, y_train)
# chech the model
print(model_svc)

# calculate the predicted value
y_prediction = model_svc.predict(x_test)

# Accessment
print('SVC--classification_report'.center(52, '-'))  # 总共打印52个字符 剧中后，其他的用-填充
print(classification_report(y_true=y_test, y_pred=y_prediction))  # support 数量
print()

print('SVC--Confusion Matrix'.center(52, '-'))
print(confusion_matrix(y_true=y_test, y_pred=y_prediction))

print("Draw!")
false_positive_rate, true_positive_rate, thresholds = roc_curve(
    y_test, y_prediction, pos_label=1)
print('false_positive_rate: ', false_positive_rate)
print('true_positive_rate: ', true_positive_rate)
print('thresholds: ', thresholds)
help
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate,
         'b', label='AUC = %0.4f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.savefig('./plot1.png', format='png')
