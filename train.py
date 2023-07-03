import joblib
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from Evaluation import *


def Read_data():
    Track_data = pd.read_csv('./Data/test/feature_test.csv')
    Track_data = pd.read_csv('./Data/feature_b.csv')
    X_train, X_val, y_train, y_val = train_test_split(
        Track_data.iloc[:, :-1], Track_data.iloc[:, -1], test_size=0.9, random_state=22)
    return X_train, X_val, y_train, y_val

X_train, X_val, y_train, y_val = Read_data()


clf = xgb.XGBClassifier(
    n_estimators=500,  # 迭代次数
    learning_rate=0.05,  # 步长
    max_depth=6,  # 树的最大深度
    min_child_weight=1,  # 决定最小叶子节点样本权重和
    subsample=1,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
    colsample_bytree=1,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
    objective="binary:logistic",
    nthread=4,
    seed=27)

# clf.fit(X_train, y_train, verbose=True, eval_metric="logloss")
# joblib.dump(clf, './Model/xgboost3.model')

sw = joblib.load('./Model/xgboost.model')
print(sw)
fit_pred = sw.predict(X_val)
Evaluation(y_val, fit_pred)


head_row = pd.read_csv('./Data/feature_b.csv', nrows=0)
print(head_row)
print(sw.feature_importances_)
