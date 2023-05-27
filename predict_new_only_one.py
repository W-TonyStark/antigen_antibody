import numpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


filename = 'D:/Sheng/sheng/Xlab/2023/antigen-antibody/单种被测物.xlsx'

# 处理excel中的数据
df = pd.read_excel(filename, sheet_name='data')
df = df.iloc[:, [0] + list(range(1, df.shape[1], 2))]
df = df.rename(columns={df.columns[0]: 'time_aaa'})
df.columns = df.columns.str.slice(stop=-4)
df_tra = df.transpose()
# 去掉时间列
df_tra = df_tra.drop(df_tra.index[0])
# 特征数据
data = np.array(df_tra)
# 数据标签
label = np.array(df_tra.transpose().columns.values.tolist())

data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2, random_state=0)
# 制作数据集
# dataset = {'data': data, 'target': label, 'frame': None, dtype='<U10'}

# 创建一个支持向量机分类器
# model = SVC()

# 创建一个决策树
# model = DecisionTreeClassifier()

# 创建一个随机森林
# model = RandomForestClassifier(max_depth=35, max_features=1, min_samples_leaf=1, min_samples_split=1, n_estimators=323, random_state=42)
# model = RandomForestClassifier()

# 创建一个K近邻
# model = KNeighborsClassifier(n_neighbors=1, p=6)
# model = KNeighborsClassifier()

# 创建一个最近中心分类器
# model = NearestCentroid()

# 创建一个MLP
model = MLPClassifier(hidden_layer_sizes=(320, 61), max_iter=1000, random_state=42)

# # 创建一个梯度提升
# model = GradientBoostingRegressor()
# # 数据标准化处理
# scaler = StandardScaler()
# data_train = scaler.fit_transform(data_train)
# data_test = scaler.fit_transform(data_test)
# # 给标签编码
# le = LabelEncoder()
# label_train = le.fit_transform(label_train)
# label_test = le.fit_transform(label_test)

# # 调参数
# param_grid = {
#     'n_neighbors': [1, 10],
#     'p': [1, 10]
# }
#
# model_grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)


# 利用贝叶斯调参数(随机森林)
pbounds = {'n_estimators': (10, 1000), 'max_depth': (1, 50),
           'min_samples_split': (1, 20), 'min_samples_leaf': (1, 10),
           'max_features': (0.1, 1.0)}

# 定义目标函数
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    clf = RandomForestClassifier(n_estimators=int(n_estimators),
                                 max_depth=int(max_depth),
                                 min_samples_split=int(min_samples_split),
                                 min_samples_leaf=int(min_samples_leaf),
                                 max_features=min(max_features, 0.999), # 避免 max_features=1.0，因为这会导致随机森林出现错误。
                                 random_state=42)
    scores = cross_val_score(clf, data_train, label_train, cv=2, scoring='accuracy')
    return np.mean(scores)

# 实例化贝叶斯优化对象，并传入目标函数和超参数空间
optimizer = BayesianOptimization(f=rf_cv, pbounds=pbounds)

# 进行优化
optimizer.maximize(init_points=10, n_iter=1000, acq='ei')

# 输出最优超参数组合和对应的分类准确率
print(optimizer.max)


# 使用训练集训练分类器
model.fit(data_train, label_train)

# 使用测试集评估分类器的性能
y_predict = model.predict(data_test)
accuracy = accuracy_score(label_test, y_predict)

print("Accuracy:", accuracy)
# print(model_grid.best_params_)